import os

vae_ckpt = '/data/hkzheng/pth/VAR/vae_ch160v4096z32.pth'  # Path to the pre-trained VAE checkpoint
var_ckpt_folder = '/data/hkzheng/pth/VAR'  # Directory containing the pre-trained VAR checkpoints

data_path = '/data/hkzheng/dataset/imagenet1k'  # Path to the ImageNet dataset

path_train_data = './output_train_data'  # Directory to save the training data for the LSRS scoring model

path_lsrs_train_save = './output_lsrs_train'  # Directory to save the output from LSRS scoring model training

path_lsrs_sample_output = './output_lsrs_run'  # Directory to save the sampling output from VAR + LSRS