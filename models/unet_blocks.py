import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicUnetBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, norm_skip=False, norm_feature=True):
        super(BasicUnetBlock, self).__init__()

        conv_in_channel = in_channels + skip_channels

        self.conv1 = self._create_conv(conv_in_channel, out_channels, 3, 1, norm_feature)
        self.conv2 = self._create_conv(out_channels, out_channels, 3, 1, norm_feature)

        if norm_skip:
            self.bn = nn.BatchNorm2d(skip_channels)
        else:
            self.bn = nn.Identity()

        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels

    def _create_conv(self, in_channel, out_channel, ks, pad, use_norm):
        l = [nn.Conv2d(in_channel, out_channel, ks, padding=pad)]
        if use_norm:
            l.append(nn.BatchNorm2d(out_channel))
        l.append(nn.ReLU())
        return nn.Sequential(*l)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        if skip is not None:
            x = torch.cat([x, self.bn(skip)], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class AttentionUnetBlock(BasicUnetBlock):
    def __init__(self, in_channels, skip_channels, out_channels, att, norm_skip=False, norm_feature=True):
        super(AttentionUnetBlock, self).__init__(in_channels, skip_channels, out_channels,
            norm_skip=norm_skip, norm_feature=norm_feature)

        if not callable(att):
            raise TypeError('Attention module must be a callable object')

        self.attention = att(in_channels, skip_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')

        if skip is not None:
            x = torch.cat([x, self.bn(skip)], dim=1)
        x = self.attention(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class AttentionGate(nn.Module):
    def __init__(self, in_channels, skip_channels):
        super(AttentionGate,self).__init__()

        self.in_G = in_channels
        self.in_L = skip_channels
        self.inner_channel = self.in_L // 2
        # G is upsampled branch
        self.W_g = nn.Sequential(
            nn.Conv2d(self.in_G, self.inner_channel, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(self.inner_channel)
            )

        # x is skip
        self.W_x = nn.Sequential(
            nn.Conv2d(self.in_L, self.inner_channel, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(self.inner_channel)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(self.inner_channel, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # SMP concat [x, skip]
        x, skip = torch.split(x, [self.in_G, self.in_L], dim=1)
        g1 = self.W_g(x)
        x1 = self.W_x(skip)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        masked = skip * psi
        out = torch.cat([x, masked], dim=1)
        return out

from timm.models.vision_transformer import Block
class TransformerGate(nn.Module):
    def __init__(self, in_channels, skip_channels, patch_size, patch_num, dim=1024, depth=2):
        super(TransformerGate, self).__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.ps = patch_size

        # compress input dimemsion ?
        # self.to_emb = None

        self.blocks = nn.ModuleList([
            Block(dim, 16, mlp_ratio=4) for _ in range(depth)
        ])
        self.emb_pos = nn.Parameter(torch.zeros(1, patch_num, dim))


    def forward(self, x):
        x, skip = torch.split(x, [self.in_channels, self.skip_channels], dim=1)
        b, c, h, w = skip.shape

        # to emb
        skip = skip.view(b, c, self.ps, h//self.ps, self.ps, w//self.ps)
        skip = skip.permute(0, 1, 2, 4, 3, 5).contiguous()
        skip = skip.view(b, c*self.ps*self.ps, -1)
        emb = skip.permute(0, 2, 1)

        # add position emb
        emb = emb + self.emb_pos

        for block in self.blocks:
            emb = block(emb)

        # to shape
        emb = emb.permute(0, 2, 1).contiguous()
        emb = emb.view(b, c, self.ps, self.ps, h//self.ps, w//self.ps)
        emb = emb.permute(0, 1, 2, 4, 3, 5).contiguous()
        emb = emb.view(b, c, h, w)

        out = torch.cat([x, emb], dim=1)

        return out
