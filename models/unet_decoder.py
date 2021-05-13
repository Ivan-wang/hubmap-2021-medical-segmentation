import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_center import CenterBlockBase
from .unet_blocks import BasicUnetBlock

class FPN(nn.Module):
    def __init__(self, input_channels:list, output_channels:list):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(in_ch, out_ch*2, kernel_size=3, padding=1),
             nn.ReLU(inplace=True), nn.BatchNorm2d(out_ch*2),
             nn.Conv2d(out_ch*2, out_ch, kernel_size=3, padding=1))
            for in_ch, out_ch in zip(input_channels, output_channels)])

    def forward(self, xs:list, last_layer):
        hcs = [F.interpolate(c(x),scale_factor=2**(len(self.convs)-i),mode='bilinear')
               for i,(c,x) in enumerate(zip(self.convs, xs))]
        hcs.append(last_layer)
        return torch.cat(hcs, dim=1)

class UnetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, block, center=None, keep_all=False):
        super(UnetDecoder, self).__init__()

        center_channel = in_channels[0]
        if center is None:
            self.center = CenterBlockBase(center_channel)
        elif callable(center):
            self.center = center(center_channel)
        else:
            raise TypeError('Center module should be a callable object')


        skip_channels = list(in_channels[1:])
        in_channels = [self.center.out_channels] + list(out_channels[:-1])

        if not callable(block):
            raise TypeError('Block module should be a callable object')

        self.blocks = nn.ModuleList()
        for stage, (i, s, o) in enumerate(zip(in_channels, skip_channels, out_channels)):
            self.blocks.append(block(i, s, o, stage))

        self.keep_all = keep_all

        self.in_channels = in_channels
        self.out_channels = [self.center.out_channels] + out_channels

    def forward(self, *features):
        # first layer is identity layer (by SMP) -- skip it
        features = features[1:]
        features = features[::-1]

        center = features[0]
        skips = features[1:]

        if self.keep_all:
            out = [self.center(center)]
            for i, decoder in enumerate(self.blocks):
                skip = skips[i] if i < len(skips) else None
                out.append(decoder(out[-1], skip))
        else:
            out = self.center(center)
            for i, decoder in enumerate(self.blocks):
                skip = skips[i] if i < len(skips) else None
                out = decoder(out, skip)

        return out

class FPNUnetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, block, fpn, center=None):
        super(FPNUnetDecoder, self).__init__()

        self.unet_decoder = UnetDecoder(in_channels, out_channels, block, center, keep_all=True)
        if not callable(fpn):
            raise TypeError('FPN object should be a callable object')

        self.fpn = fpn(self.unet_decoder.out_channels[:-1])

    def forward(self, *features):
        decoder_features = self.unet_decoder(*features)
        out = self.fpn(decoder_features[:-1], decoder_features[-1])
        return out

