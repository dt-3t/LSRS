import os
import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import build_vae
from utils.data import build_dataset

from configs import *

def main():
    save_dir = path_train_data
    batch_size = 50

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(save_dir, exist_ok=True)
    
    _, train_set, val_set = build_dataset(
        data_path=data_path,
        final_reso=256,
        hflip=False 
    )
    
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    vae = build_vae(device=device)
    vae_state = torch.load(vae_ckpt, map_location='cpu')
    vae.load_state_dict(vae_state)
    vae.eval()
    
    print("Processing training set...")
    data_list = [[] for _ in range(1000)] 
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Processing batches")):
        images = images.to(device)
        B = images.shape[0]
        
        with torch.no_grad():
            indices = vae.img_to_idxBl(images)  # list
            indices = torch.cat(indices, dim=1)  # (B, L)
        unique_labels = torch.unique(labels)
        for c in unique_labels:
            c = c.item()
            mask = (labels == c)
            class_indices = indices[mask] 
            class_indices = class_indices.cpu()
            if data_list[c] == []:
                data_list[c] = class_indices
            else:
                data_list[c] = torch.cat([data_list[c], class_indices], dim=0)

    torch.save(data_list, os.path.join(save_dir, 'gt_indices.pth'))
    print(f"Saved VAE indices to {os.path.join(save_dir, 'gt_indices.pth')}")

if __name__ == "__main__":
    main()