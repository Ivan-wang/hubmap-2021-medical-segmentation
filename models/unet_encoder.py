import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class BasicUnetEncoder(nn.Module):
    def __init__(self, n_channels):
        super(BasicUnetEncoder, self).__init__()

        self.out_channels = [n_channels, 64, 128, 256, 512, 1024]
        self.blocks = nn.ModuleList()

        for i in range(1, len(self.out_channels)):
            if i == 1:
                self.blocks.append(DoubleConv(self.out_channels[i-1], self.out_channels[i]))
            else:
                self.blocks.append(Down(self.out_channels[i-1], self.out_channels[i]))

    def forward(self, x):
        features = [x]
        for b in self.blocks:
            features.append(b(features[-1]))

        return features

import timm
from timm.models.layers import trunc_normal_
class ViTUnetEncoder(nn.Module):
    def __init__(self):
        super(ViTUnetEncoder, self).__init__()

        vit = timm.create_model('vit_base_resnet50_224_in21k', pretrained=True)

        self.out_channels = [3, 64, 256, 512, 768]

        patch_embed = vit.patch_embed
        backbone = patch_embed.backbone

        self.blocks = nn.ModuleList([
            nn.Sequential(backbone.stem.conv, backbone.stem.norm),
            # backbone.stem, # 64, 64
            nn.Sequential(backbone.stem.pool, backbone.stages[0]),
            # backbone.stages[0], # 64, 64
            backbone.stages[1], # 512, 32
            backbone.stages[2], # 1024, 16
        ])

        self.center = nn.Sequential()
        self.to_embed = nn.Sequential(
            backbone.norm,
            backbone.head,
            patch_embed.proj
        )

        # self.emb_pos = nn.Parameter(torch.zeros(1, 256, 768))
        self.emb_pos = nn.Parameter(torch.zeros(1, 1024, 768))
        self.emb_drop = nn.Dropout(p=0.0)
        self.transformer = vit.blocks

        trunc_normal_(self.emb_pos, 0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'emb_pos'}

    def forward(self, x):
        features = [x]

        for b in self.blocks:
            features.append(b(features[-1]))

        emb = self.to_embed(features[-1])
        emb = emb.flatten(2).transpose(1, 2)

        # add pos_embed here
        emb = emb + self.emb_pos
        emb = self.emb_drop(emb)

        # to transformer
        for b in self.transformer:
            emb = b(emb)

        features.pop()
        features.append(emb)

        return features


if __name__ == '__main__':
    m = ViTUnetEncoder()
    x = torch.randn(1, 3, 256, 256)
    out = m(x)
    print([o.shape for o in out])
