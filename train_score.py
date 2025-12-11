import os
import os.path as osp
import argparse
import json
import math
import random
from datetime import datetime
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image as PImage
from tqdm import tqdm
from einops import rearrange
from score_net import RewardNet
from models import build_vae

from configs import *

class SampleDataset(Dataset):
    def __init__(self, loaded_samples):
        self.loaded_samples = loaded_samples
        self.n_d = len(self.loaded_samples)
        assert self.n_d == 2

        self.dataset_len_prefix_sum = [0] * loaded_samples[0].n_class
        for i, (k_per_class_0, k_per_class_1) in enumerate(zip(self.loaded_samples[0].k_each_class, self.loaded_samples[1].k_each_class)):
            self.dataset_len_prefix_sum[i] = k_per_class_0 * k_per_class_1
            if i-1 >= 0:
                self.dataset_len_prefix_sum[i] += self.dataset_len_prefix_sum[i-1]
        self.dataset_len_prefix_sum = torch.tensor(self.dataset_len_prefix_sum, dtype=torch.int64)

    def __len__(self):
        return self.dataset_len_prefix_sum[-1]

    def __getitem__(self, idx):
        class_idx = torch.searchsorted(self.dataset_len_prefix_sum, idx, right=True)
        class_k_1 = self.loaded_samples[1].k_each_class[class_idx]
        reminder = idx - self.dataset_len_prefix_sum[class_idx-1] if class_idx-1 >= 0 else idx
        pos_0 = reminder // class_k_1
        pos_1 = reminder % class_k_1
        return torch.stack([self.loaded_samples[0].data[class_idx][pos_0, :], 
                self.loaded_samples[1].data[class_idx][pos_1, :]]), class_idx

def split_oned_dataset(oned_dataset, ratio1=0.5, ratio2=0.5, random_seed=None):
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
    
    k_each_class = oned_dataset.k_each_class
    list_1 = []
    list_2 = []
    
    for idx_class, k_class in enumerate(k_each_class):
        indices = torch.randperm(k_class)

        split1 = int(k_class * ratio1)
        split2 = split1 + int(k_class * ratio2)

        subset1_indices = indices[:split1]
        subset2_indices = indices[split1:split2]

        subset1_data = oned_dataset.data[idx_class][subset1_indices]
        subset2_data = oned_dataset.data[idx_class][subset2_indices]
        
        list_1.append(subset1_data)
        list_2.append(subset2_data)
    
    return (
        OnedDataset(list_1, oned_dataset.name + '_subset1', rank=oned_dataset.rank, max_k_per_class=oned_dataset.max_k_per_class),
        OnedDataset(list_2, oned_dataset.name + '_subset2', rank=oned_dataset.rank, max_k_per_class=oned_dataset.max_k_per_class)
    )


class OnedDataset():
    def __init__(self, data, name, rank, n_class=1000, max_k_per_class=100):
        self.max_k_per_class = max_k_per_class
        self.data = [d[:max_k_per_class].to(torch.int32) for d in data] 
        self.name = name
        self.rank = rank
        self.n_class = n_class
        self.k_each_class = [self.data[i].shape[0] for i in range(self.n_class)]

def random_split_dataset(dataset, ratio=1.0, random_seed=None):
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
    
    subset_size = int(len(dataset) * ratio)
    _, subset = torch.utils.data.random_split(
        dataset,
        [len(dataset) - subset_size, subset_size]
    )
    return subset

def load_samples(file_paths, val_retio=0.2, max_k_per_class=100, random_seed=None, sub_train_ratio=1.0):
    train_samples = []
    val_samples = []
    
    for i, path in enumerate(file_paths):
        file_name = osp.basename(path)

        data = torch.load(path)
        oned_dataset = OnedDataset(data, file_name, i, max_k_per_class=max_k_per_class)

        train_dataset, val_dataset = split_oned_dataset(oned_dataset, ratio1=(1-val_retio)*sub_train_ratio, ratio2=val_retio, random_seed=random_seed)
        train_samples.append(train_dataset)
        val_samples.append(val_dataset)
    
    return train_samples, val_samples

def fhat_to_save(vae, fhat, save_folder='./tmp_imgs', suffix=''):
    os.makedirs(save_folder, exist_ok=True) 
    img = vae.fhat_to_img(fhat).add_(1).mul_(0.5)
    img = torchvision.utils.make_grid(img, nrow=8, padding=0, pad_value=1.0)
    img = img.permute(1, 2, 0).mul(255).cpu().numpy()
    img = PImage.fromarray(img.astype(np.uint8))
    img.save(os.path.join(save_folder, f"img_{suffix}.png"))

def logsigmoid_rank_loss(positive_scores, negative_scores, reduction='mean'):
    score_diffs = positive_scores - negative_scores
    losses = -F.logsigmoid(score_diffs)
    
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.mean(losses)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

def rank_loss(scores, mean_batch=False):
    # scores: (n_stage*B, n)
    _, n = scores.shape
    assert n == 2
    if mean_batch:
        scores = rearrange(scores, '(n_stage B) n -> n_stage B n', n_stage=n_stage)
        scores = scores.mean(dim=1)  # (n_stage, n)
    positive_scores = scores[:, 1]
    negative_scores = scores[:, 0]
    loss = logsigmoid_rank_loss(positive_scores, negative_scores)
    return loss

def compu_acc(scores, n_stage=None):
    # scores: (n_stage*B, n)
    if n_stage is not None:
        scores = rearrange(scores, '(n_stage B) n -> n_stage B n', n_stage=n_stage)
        stage_acc = (scores[:, :, 1] > scores[:, :, 0]).float().mean(dim=1)  # (n_stage,)
        return stage_acc
    else:
        acc_now = (scores[:, 1] > scores[:, 0]).float().mean().item()
        return acc_now

SEED = 42 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train depth prediction model')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=4, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default=path_lsrs_train_save, help='Model save directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Training device')
    parser.add_argument('--d', type=int, default=16, help='Data dimension parameter d')
    parser.add_argument('--sub_train_ratio', type=float, default=1.0, help='Sub-sampling ratio for training data')
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_save_dir = os.path.join(args.save_dir, f"run_score_d{args.d}_{timestamp}")
    os.makedirs(run_save_dir, exist_ok=True)
    args.save_dir = run_save_dir

    file_paths = [
        path_train_data + f'/samples_idx_d{args.d}_n1000.pth', 
         os.path.join(path_train_data, 'gt_idx.pth'),
    ]

    print("="*50)
    print("args:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("="*50)

    log_file = os.path.join(args.save_dir, 'training_log.json')
    log_data = {
        'args': vars(args),
        'epochs': [],
        'best_val_loss': float('inf'),
        'best_val_acc': -1,
        'file_paths': file_paths
    }

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    
    patch_nums = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
    stage_len = [i**2 for i in patch_nums]
    assert sum(stage_len) == 680
    n_stage = len(patch_nums)

    val_retio = 0.1
    train_samples, val_samples = load_samples(file_paths, val_retio=val_retio, max_k_per_class=10000, 
                                              random_seed=SEED, sub_train_ratio=args.sub_train_ratio)

    dataset_train = SampleDataset(train_samples)
    dataset_val = SampleDataset(val_samples)

    print('using sub data...')
    dataset_train = random_split_dataset(dataset_train, ratio=0.01, random_seed=SEED)
    dataset_val = random_split_dataset(dataset_val, ratio=0.1, random_seed=SEED)

    train_dataloader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    stage_label = torch.arange(n_stage, dtype=torch.long, device=device)  # (n_stage,)

    vae = build_vae(device=device)
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    vae.eval()
    for p in vae.parameters(): p.requires_grad_(False)

    score_net = RewardNet()

    score_net = score_net.to(device)
    
    optimizer = torch.optim.Adam(score_net.parameters(), lr=args.lr)

    warmup_steps = len(train_dataloader)
    total_steps = len(train_dataloader) * args.epochs
    print(f"total_steps: {total_steps}, warmup_steps: {warmup_steps}")

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    total_params = sum(p.numel() for p in score_net.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params/1e6:.2f}M")

    best_val_loss = float('inf')
    best_val_acc = -1
    print('Starting training...')
    global_step = 0
    for epoch in range(args.epochs):
        score_net.train()
        train_loss = 0.0

        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        smooth_loss = None 
        for idx, (batch_data, class_idx) in enumerate(train_pbar):
            batch_data = batch_data.to(device)  # (B, n_deep, L)
            B, n_deep, L = batch_data.shape
            class_idx = class_idx.to(device)  # (B,)
            optimizer.zero_grad()

            batch_data = rearrange(batch_data, 'B n_deep L -> (B n_deep) L')
            batch_data_stage = torch.split(batch_data, stage_len, dim=1)
            x_list = vae.quantize.idxBl_to_fhat(batch_data_stage)
            x = torch.cat(x_list, dim=0)  # (n_stage*B*n_deep, C, H, W)
            x = x.detach()

            # x_tmp = rearrange(x, '(n_stage B n_deep) C H W -> n_stage B n_deep C H W', n_stage=n_stage, B=B, n_deep=n_deep)
            # fhat_to_save(vae, x_tmp[-1, :32, 0], suffix=f"epoch_{epoch}_step_{idx}_deep_0")
            # fhat_to_save(vae, x_tmp[-1, :32, 1], suffix=f"epoch_{epoch}_step_{idx}_deep_1")

            stage_idx = stage_label[:, None, None].repeat(1, B, n_deep)
            stage_idx = rearrange(stage_idx, 'n_stage B n_deep -> (n_stage B n_deep)')
            class_idx = class_idx[None, :, None].repeat(n_stage, 1, n_deep)
            class_idx = rearrange(class_idx, 'n_stage B n_deep -> (n_stage B n_deep)')

            scores = score_net(x, class_idx, stage_idx)  # (n_stage*B*n_deep, 1)
            scores = rearrange(scores, '(n_stage B n_deep) 1 -> (n_stage B) n_deep', n_stage=n_stage, B=B, n_deep=n_deep)

            loss = rank_loss(scores)

            smooth_loss = loss.item() if smooth_loss is None else 0.1 * loss.item() + 0.9 * smooth_loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            train_loss += loss.item()

            train_pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Smooth Loss': f"{smooth_loss:.4f}",
                'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        score_net.eval()
        val_loss = 0.0
        val_acc = 0.0
        all_scores = []
        all_stage_acc = torch.zeros(n_stage, device=device)
        
        val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        all_outputs = [] 
        with torch.no_grad():
            for batch_data, class_idx in val_pbar: 
                batch_data = batch_data.to(device)
                class_idx = class_idx.to(device)
                B, n_deep, L = batch_data.shape

                batch_data = rearrange(batch_data, 'B n_deep L -> (B n_deep) L')
                batch_data_stage = torch.split(batch_data, stage_len, dim=1)
                x_list = vae.quantize.idxBl_to_fhat(batch_data_stage)
                x = torch.cat(x_list, dim=0)  # (n_stage*B*n_deep, C, H, W)
                x = x.detach()

                stage_idx = stage_label[:, None, None].repeat(1, B, n_deep)
                stage_idx = rearrange(stage_idx, 'n_stage B n_deep -> (n_stage B n_deep)')
                class_idx = class_idx[None, :, None].repeat(n_stage, 1, n_deep)
                class_idx = rearrange(class_idx, 'n_stage B n_deep -> (n_stage B n_deep)')

                scores = score_net(x, class_idx, stage_idx)  # (n_stage*B*n_deep, 1)
                scores = rearrange(scores, '(n_stage B n_deep) 1 -> (n_stage B) n_deep', n_stage=n_stage, B=B, n_deep=n_deep)

                scores_save = rearrange(scores, '(n_stage B) n_deep -> n_stage B n_deep', n_stage=n_stage, B=B, n_deep=n_deep)
                class_idx_save = rearrange(class_idx, '(n_stage B n_deep) -> n_stage B n_deep', n_stage=n_stage, B=B, n_deep=n_deep)
                batch_output = {
                    'scores': scores_save.cpu().numpy(),
                    'class_idx': class_idx_save.cpu().numpy()
                }
                all_outputs.append(batch_output)

                # scores_tmp = rearrange(scores, '(n_stage B) n_deep -> n_stage B n_deep', n_stage=n_stage, B=B, n_deep=n_deep)
                # scores_tmp = scores_tmp.mean(dim=1)
                acc_now = compu_acc(scores)
                val_acc += acc_now

                stage_acc = compu_acc(scores, n_stage=n_stage)  # (n_stage,)
                all_stage_acc += stage_acc

                batch_loss = rank_loss(scores)
                val_loss += batch_loss.item()
                all_scores.append(scores.cpu().numpy())
                val_pbar.set_postfix({
                    'Val Loss': f"{batch_loss.item():.4f}",
                    'Val Acc': f"{acc_now:.4f}" 
                })

        outputs_path = osp.join(args.save_dir, f'val_outputs_epoch_{epoch+1}.npz')
        np.savez(outputs_path, *all_outputs)
        print(f'Validation outputs saved to: {outputs_path}')

        all_scores = np.concatenate(all_scores, axis=0).flatten()
        scores_path = osp.join(args.save_dir, f'scores_epoch_{epoch+1}.npy')
        np.save(scores_path, all_scores)
        plt.figure(figsize=(10, 6))
        plt.hist(all_scores, bins=200, alpha=0.7) 
        plt.title(f'Scores Distribution - Epoch {epoch+1}')
        plt.xlabel('Score Value')
        plt.ylabel('Frequency')
        plot_path = osp.join(args.save_dir, f'scores_dist_epoch_{epoch+1}.png')
        plt.savefig(plot_path)
        plt.close()

        train_loss /= len(train_dataloader)
        val_loss /= len(val_dataloader)
        val_acc /= len(val_dataloader)
        all_stage_acc /= len(val_dataloader)
        stage_acc_list = all_stage_acc.cpu().tolist()
        
        print("\n" + "="*50)
        print(f"Epoch: {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}") 
        print("="*50 + "\n")
        
        model_path = osp.join(args.save_dir, f'model_epoch_{epoch+1}.pth')
        torch.save(score_net.state_dict(), model_path)
        print(f'Model weights saved to: {model_path}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = osp.join(args.save_dir, 'best_loss_model.pth')
            torch.save(score_net.state_dict(), best_model_path)
            print(f'Best loss weights saved to: {best_model_path}')
            log_data['best_val_loss'] = best_val_loss
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_acc_model_path = osp.join(args.save_dir, 'best_acc_model.pth')
            torch.save(score_net.state_dict(), best_acc_model_path)
            print(f'Best acc weights saved to: {best_acc_model_path}')
            log_data['best_val_acc'] = best_val_acc

        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'val_acc': float(val_acc),
            'stage_acc': stage_acc_list,  
            'lr': float(optimizer.param_groups[0]['lr']),
            'global_step': global_step
        }
        log_data['epochs'].append(epoch_log)
            
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    print('Training completed!')
