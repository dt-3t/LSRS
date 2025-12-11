from tqdm import tqdm
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import tempfile
import shutil
import PIL.Image as PImage
import time
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed

from models import VQVAE, build_vae_var
from load_score_models import load_score_net
import argparse

from configs import *

def parse_args():
    parser = argparse.ArgumentParser(description='VAR model sampling script')
    parser.add_argument("-b", "--batch_size", type=int, default=80,
                       help="batch size for sampling")
    parser.add_argument("-d", "--model_depth", type=int, default=16, choices=[16, 20, 24, 30], 
                       help="model depth, options: 16, 20, 24, 30")
    parser.add_argument("-n", "--npz_name", type=str, default='images',
                       help="name for generated .npz file")
    parser.add_argument("-u", "--use_discriminator", action='store_true',
                       help="enable discriminator (requires score_net)")
    parser.add_argument("-s", "--score_net_path", type=str, default='',
                       help="path to score_net model")
    
    # st <= scale_num < ed will use LSRS sampling
    parser.add_argument("--st", type=int, default=1,
                       help="LSRS parameter st (default: 1), options: 0~9")
    parser.add_argument("--ed", type=int, default=10,
                          help="LSRS parameter ed (default: 10), options: 1~10, must be strictly greater than st")
    parser.add_argument("--mk", type=int, default=4,
                       help="LSRS parameter m_k (default: 4)")
    
    parser.add_argument("--top_k_lsrs", "-tkl", type=int, default=1,
                       help="top-k for LSRS sampling (optional, fallback to global top_k)")
    parser.add_argument("--temperature_lsrs", "-temp", type=float, default=1.0,
                       help="temperature for LSRS sampling (default: 1.0)")
    
    parser.add_argument('--folder_suffix', type=str, default='',
                       help='suffix for output folder name to distinguish different runs')

    args = parser.parse_args()
    
    if args.use_discriminator and not args.score_net_path:
        parser.error("score_net_path (-s/--score_net_path) is required when using discriminator")
    
    return args


def create_npz_from_sample_folder(sample_folder: str, npz_name: str):
    import os, glob
    import numpy as np
    from tqdm import tqdm
    from PIL import Image

    samples = []
    pngs = glob.glob(os.path.join(sample_folder, '*.png')) + glob.glob(os.path.join(sample_folder, '*.PNG'))
    assert len(pngs) == 50_000, f'{len(pngs)} png files found in {sample_folder}, but expected 50,000'
    for png in tqdm(pngs, desc='Building .npz file from samples (png only)'):
        with Image.open(png) as sample_pil:
            sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (50_000, samples.shape[1], samples.shape[2], 3)
    npz_path = f'{npz_name}.npz'
    np.savez(npz_path, arr_0=samples)
    print(f'Saved .npz file to {npz_path} [shape={samples.shape}].')

args = parse_args()

var_ckpt = os.path.join(var_ckpt_folder, f'var_d{args.model_depth}.pth')
MODEL_DEPTH = args.model_depth
assert MODEL_DEPTH in {16, 20, 24, 30}

score_net_type = 'score' if 'score' in args.score_net_path.lower() else 'class' if 'class' in args.score_net_path.lower() else 'unknown'
if not args.use_discriminator:
    output_dir_base = os.path.join(path_lsrs_sample_output, f'd{args.model_depth}')
else:
    lsrs_parts = [f'd{args.model_depth}', f'{score_net_type}', f'st{args.st}', f'ed{args.ed}', f'mk{args.mk}']
    if args.top_k_lsrs > 1:
        lsrs_parts.append(f'topk{args.top_k_lsrs}')
        lsrs_parts.append(f'temp{args.temperature_lsrs:.3g}'.replace('.', 'p'))  # e.g., 0.95 -> temp0p95
    lsrs_parts.append(args.folder_suffix)
    output_dir_base = os.path.join(path_lsrs_sample_output, '_'.join(lsrs_parts))
output_dir = output_dir_base
counter = 1
while os.path.exists(output_dir):
    output_dir = f"{output_dir_base}_{counter}"
    counter += 1
os.makedirs(output_dir, exist_ok=True)
output_path = tempfile.mkdtemp(prefix='var_images_', dir=output_dir)

patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
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

# LSRS parameters
st = args.st
ed = args.ed
m_k = args.mk
score_net_path = os.path.expanduser(args.score_net_path)
discriminator = load_score_net(score_net_path, device) if args.use_discriminator else None
seed = 2
j_plus = 3000

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')

with open(osp.join(output_dir, 'params.txt'), 'w') as f:
    f.write(f"batch_size: {args.batch_size}\n")
    f.write(f"model_depth: {args.model_depth}\n")
    f.write(f"npz_name: {args.npz_name}\n")
    f.write(f"seed: {seed}\n")
    f.write(f"j_plus: {j_plus}\n")
    f.write(f"use_discriminator: {args.use_discriminator}\n")
    if args.use_discriminator:
        f.write(f"score_net_path: {score_net_path}\n")
        f.write(f"st: {st}\n")
        f.write(f"ed: {ed}\n")
        f.write(f"m_k: {m_k}\n")
        f.write(f"top_k_lsrs: {args.top_k_lsrs}\n")
        f.write(f"temperature_lsrs: {args.temperature_lsrs}\n")
        shutil.copy2(score_net_path, osp.join(output_dir, osp.basename(score_net_path)))

top_k = 900
cfg = 1.5
top_p = 0.96
more_smooth = False
batch_size = args.batch_size

def rollback_and_claen(output_path):
    import os

    if not osp.exists(output_path):
        os.makedirs(output_path)
    else:
        png_files = [f for f in os.listdir(output_path) if f.endswith('.png') or f.endswith('.PNG')]
        if png_files:
            print(f"Found {len(png_files)} existing PNG files in output directory: {output_path}")
            choice = input("Please choose an option:\n1. Resume from breakpoint\n2. Start over\nEnter your choice (1 or 2): ")
            while choice not in ['1', '2']:
                choice = input("Invalid input, please enter 1 or 2: ")
            if choice == '2':
                for f in png_files:
                    os.remove(osp.join(output_path, f))
                print("All existing images have been removed, starting fresh generation")
                return 0
            else:
                print("Resuming generation from breakpoint")

    max_index = -1
    class_num_count = {}
    for filename in os.listdir(output_path):
        if filename.endswith('.PNG') or filename.endswith('.png'):
            filename_pure = os.path.splitext(filename)[0]
            class_index = int(filename_pure.split('_')[0])
            if class_index not in class_num_count:
                class_num_count[class_index] = 0
            class_num_count[class_index] += 1

    for class_index, num_count in class_num_count.items():
        if num_count == 50:
            max_index = max(max_index, class_index)
    print(f"max full class index: {max_index}")

    for filename in os.listdir(output_path):
        if filename.endswith('.PNG') or filename.endswith('.png'):
            filename_pure = os.path.splitext(filename)[0]
            class_index = int(filename_pure.split('_')[0])
            if class_index > max_index:
                os.remove(os.path.join(output_path, filename))

    return max_index + 1

st_class = rollback_and_claen(output_path)
print(f"start class index:{st_class}")

print("#####################sampling begin!#####################")
for i in range(st_class, 1000, batch_size):
    if i >= 1000:
        break
    batch_ed = min(i + batch_size - 1, 999)
    print("class index:{} ~ {}".format(i, batch_ed))
    class_labels = range(i, batch_ed + 1)
    for j in tqdm(range(50)):  # 50 samples per class
        with torch.inference_mode():
            B = len(class_labels)  # batch size real
            label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
            with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
                if discriminator is not None:
                    recon_B3HW = var.autoregressive_infer_cfg_lsrs(
                        B=B, label_B=label_B,
                        cfg=cfg, top_k=top_k, top_p=top_p,
                        g_seed=j + j_plus,
                        more_smooth=more_smooth,
                        discriminator=discriminator,
                        st=st, ed=ed, m_k=m_k,
                        top_k_lsrs=args.top_k_lsrs,
                        temperature_lsrs=args.temperature_lsrs
                    )
                else:
                    recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B,
                        cfg=cfg, top_k=top_k, top_p=top_p, g_seed=j+j_plus, more_smooth=False
                    )
            # recon_B3HW: torch.Tensor(B, 3, H, W)
            for idx in range(B):
                single_image = recon_B3HW[idx]  # (3, H, W)
                single_image = single_image.mul_(255).permute(1, 2, 0).cpu().numpy()
                single_image = PImage.fromarray(single_image.astype(np.uint8))
                single_image.save(output_path + "/" + str(i + idx) + "_" + str(j) + ".PNG")

create_npz_from_sample_folder(output_path, osp.join(output_dir, args.npz_name))
try:
    shutil.rmtree(output_path)
    print(f"Temporary folder deleted: {output_path}")
except Exception as e:
    print(f"Failed to delete temporary folder: {e}")

print("#####################sampling completed!#####################")
