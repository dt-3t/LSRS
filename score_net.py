import torch
import torch.nn as nn
from einops import rearrange

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, in_HW, stride=1):
        super(ResidualBlock, self).__init__()
        out_HW = in_HW // stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.ln1 = nn.LayerNorm([out_channels, out_HW, out_HW])
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.ln2 = nn.LayerNorm([out_channels, out_HW, out_HW])
        
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.ln3 = nn.LayerNorm([out_channels, out_HW, out_HW])

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.LayerNorm([out_channels, out_HW, out_HW])
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.ln1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.ln2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.ln3(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class RewardNet(nn.Module):
    def __init__(self):
        super(RewardNet, self).__init__()

        dropout_ratio = 0.05

        self.conv_stage_1 = nn.Sequential(
            ResidualBlock(32, 64, in_HW=16, stride=2),
            ResidualBlock(64, 64, in_HW=8),
            nn.Dropout(dropout_ratio)
        )

        self.conv_stage_2 = nn.Sequential(
            ResidualBlock(64, 128, in_HW=8, stride=2),
            ResidualBlock(128, 128, in_HW=4),
            nn.Dropout(dropout_ratio)
        )

        self.conv_stage_3 = nn.Sequential(
            ResidualBlock(128, 256, in_HW=4, stride=2),
            ResidualBlock(256, 256, in_HW=2),
            nn.Dropout(dropout_ratio)
        )

        self.class_label_embedding = nn.Embedding(1000, 128)
        self.class_label_proj = nn.Linear(128, 128)

        self.stage_label_embedding = nn.Embedding(10, 128)
        self.stage_label_proj = nn.Linear(128, 128)

        self.flatten = nn.Flatten()
        self.fuser = nn.Sequential(
            nn.Linear(256 * 2 * 2 + 256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2), 
            nn.Linear(256, 128)
        )
        self.out_proj = nn.Linear(128, 1)
        self._initialize_weights()

    def forward(self, x, class_labels, stage_label):
        # x: (bs, 32, h, w) , h = w = 16
        # labels: (bs,)

        x = self.conv_stage_1(x)  # (bs, 64, h/2, w/2)

        x = self.conv_stage_2(x)  # (bs, 128, h/4, w/4)

        x = self.conv_stage_3(x)  # (bs, 256, h/8, w/8)

        x = self.flatten(x)  # (bs, 256 * 2 * 2)

        class_label_feat = self.class_label_embedding(class_labels)  # (bs, 128)
        class_label_feat = self.class_label_proj(class_label_feat)  # (bs, 128)

        stage_label_feat = self.stage_label_embedding(stage_label)  # (bs, 128)
        stage_label_feat = self.stage_label_proj(stage_label_feat)  # (bs, 128)

        x = torch.cat([x, class_label_feat, stage_label_feat], dim=1)  # (bs, 256 * 2 * 2 + 256)

        x = self.fuser(x)  # (bs, 128)

        scores = self.out_proj(x)  # (bs, 1)

        return scores

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, mean=0, std=0.02) 
            elif isinstance(m, nn.Parameter):
                if m.dim() > 1:
                    nn.init.trunc_normal_(m, std=0.02)