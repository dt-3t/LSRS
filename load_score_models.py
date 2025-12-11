import torch
import os
from models import build_vae
from score_net import RewardNet

from configs import *

def load_vae(device):
    vae = build_vae(device=device)
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae

def load_score_net(score_net_path, device):
    score_net = RewardNet()
    score_net.load_state_dict(torch.load(score_net_path, map_location='cpu'))
    score_net = score_net.to(device)
    score_net.eval()
    return score_net
