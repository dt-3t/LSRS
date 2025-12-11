from tqdm import tqdm
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage
import shutil

setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

from models import VQVAE, build_vae_var
import argparse
from configs import *

def trans_to_list(data):
    k, b, l = data.shape
    a_list = [[] for _ in range(k)]
    for i in range(k):
        a_list[i] = data[:, i]

    return a_list

def parse_args():
    parser = argparse.ArgumentParser(description='VAR model sampling script')
    parser.add_argument("--batch_size", type=int, default=40,
                       help="sampling batch size")
    parser.add_argument("--model_depth", type=int, default=16, choices=[16, 20, 24, 30], 
                       help="model depth, options: 16, 20, 24, 30")
    parser.add_argument("--k_sample", type=int, default=1000, 
                       help="number of samples per class")
    return parser.parse_args()

args = parse_args()

seed = 0
top_k = 900
cfg = 1.5
top_p = 0.96
k_sample_per_class = args.k_sample 

MODEL_DEPTH = args.model_depth
assert MODEL_DEPTH in {16, 20, 24, 30}
var_ckpt = os.path.join(var_ckpt_folder, f'var_d{MODEL_DEPTH}.pth')
vae_ckpt = os.path.expanduser(vae_ckpt)
var_ckpt = os.path.expanduser(var_ckpt)
output_path = path_train_data
os.makedirs(output_path, exist_ok=True) 

print(f"\033[32m>>>> MODEL_DEPTH: {MODEL_DEPTH} <<<<\033[0m")

patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
L = sum([p * p for p in patch_nums])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'vae' not in globals() or 'var' not in globals():
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
    )

vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=False)
vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')

batch_size = args.batch_size

print(f">>>> start <<<<")
sample_all_list = []
for i in range(0, 1000, batch_size):
    if i >= 1000:
        break
    ed = min(i + batch_size - 1, 999)
    print("class index:{} ~ {}".format(i, ed))
    class_labels = range(i, ed + 1)
    B = len(class_labels)  # batch size real
    label_B: torch.LongTensor = torch.tensor(class_labels, device=device)  # (B,)
    sample_list = []
    for j in tqdm(range(k_sample_per_class)):  # k samples per class
        with torch.inference_mode():
            with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
                idx_list = var.autoregressive_infer_cfg_idx(B=B, label_B=label_B,
                cfg=cfg, top_k=top_k, top_p=top_p, g_seed=j, more_smooth=False)
            idx_cat = torch.cat(idx_list, dim=1)  # (B, L)
            assert idx_cat.shape == (B, L)
            assert idx_cat.min() >= 0 and idx_cat.max() < 4096
            idx_cat = idx_cat.to(torch.int16).cpu()  # to CPU
        sample_list.append(idx_cat)
    sample_cat = torch.stack(sample_list, dim=0)  # (k, B, L)
    sample_all_list.append(sample_cat)

print(f">>>> end <<<<")

output_file = osp.join(output_path, f'samples_idx_d{MODEL_DEPTH}_n{k_sample_per_class}.pt')

sample_all_cat = torch.cat(sample_all_list, dim=1)  # (k, n_class, L)
assert sample_all_cat.shape == (k_sample_per_class, 1000, L)
sample_all_list = trans_to_list(sample_all_cat)

torch.save(sample_all_list, output_file)
print(f">>>> save to: {output_file} <<<<")










