import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationModel
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.encoders import get_encoder
import torch.nn.functional as F

from .unet_center import BasicCenterBlock, ASPPCenterBlock, CenterBlockBase
from .unet_blocks import BasicUnetBlock
from .unet_decoder import UnetDecoder, FPNUnetDecoder, FPN

class UnetBase(SegmentationModel):
    def __init__(self, encoder, decoder, segmentation_head=None, drop=0.1, **kwargs):
        super(UnetBase, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        if drop is not None:
            self.drop = nn.Dropout2d(drop)
        else:
            self.drop = nn.Identity()

        if segmentation_head is not None:
            self.segmentation_head = segmentation_head
        else:
            self.segmentation_head = SegmentationHead(
                self.decoder.out_channels,
                **kwargs
            )

        self.classification_head = None

        self.name = 'u-base'
        self.initialize()

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.drop(self.decoder(*features))

        masks = self.segmentation_head(decoder_output)

        return masks

from .unet_encoder import BasicUnetEncoder
def BasicUnet(n_channels, n_classes=1, use_inter_head=False):
    encoder = BasicUnetEncoder(n_channels)
    in_channels = encoder.out_channels

    # first channel is always the image channel -- remove it
    in_channels = in_channels[1:]
    in_channels = in_channels[::-1]

    center = lambda in_c: BasicCenterBlock(in_c, compress=2)
    block = lambda in_c, skip_c, out_c, stage: BasicUnetBlock(in_c, skip_c, out_c,
        norm_skip=False, norm_feature=True)

    in_channels = list(in_channels)
    out_channels = [256, 128, 64, 32]
    head = SegmentationHead(out_channels[-1], out_channels=n_classes, upsampling=1)

    decoder = UnetDecoder(in_channels, out_channels, block, center)

    unet = UnetBase(encoder, decoder, head)

    return unet

def BasicUnetRes50(norm_skip=False, norm_feature=True, use_inter_head=False):
    encoder = get_encoder(
        'resnet50', weights='imagenet', in_channels=3
    )
    in_channels = encoder.out_channels

    # first channel is always the image channel -- remove it
    in_channels = in_channels[1:]
    in_channels = in_channels[::-1]

    # compress 2 / 4
    center = lambda in_c: BasicCenterBlock(in_c, compress=4)
    block = lambda in_c, skip_c, out_c, stage: BasicUnetBlock(in_c, skip_c, out_c,
        norm_skip=norm_skip, norm_feature=norm_feature)

    # out_channels = [512, 256, 128, 32] # 1/2 of skip channels
    if use_inter_head:
        out_channels = [256, 128, 64, 32]  # 1/4 of skip channels
        decoder = UnetDecoder(in_channels, out_channels, block, center)
        head = SegmentationHead(out_channels[-1], out_channels=9, upsampling=2)
    else:
        in_channels = list(in_channels) + [0]
        out_channels = [256, 128, 64, 32]  # 1/4 of skip channels
        out_channels += [16]
        decoder = UnetDecoder(in_channels, out_channels, block, center)
        head = SegmentationHead(out_channels[-1], out_channels=9, upsampling=1)

    unet = UnetBase(encoder, decoder, head)

    return unet

def FPNUnetSE50():
    encoder = get_encoder(
        'se_resnet50', weights='imagenet'
    )
    in_channels = encoder.out_channels

    # first channel is always the image channel -- remove it
    in_channels = in_channels[1:]
    in_channels = in_channels[::-1]

    center = lambda in_c: BasicCenterBlock(in_c, compress=4)
    block = lambda in_c, skip_c, out_c, stage: BasicUnetBlock(in_c, skip_c, out_c)
    fpn = lambda channels: FPN(channels, [16]*4)

    out_channels = [256, 128, 64, 32]
    decoder = FPNUnetDecoder(in_channels, out_channels, block, fpn, center)

    head = SegmentationHead(out_channels[-1]+16*4, out_channels=1, upsampling=2)

    unet = UnetBase(encoder, decoder, head)

    return unet

def ASPPUnetSE50():
    encoder = get_encoder(
        'se_resnet50', weights='imagenet'
    )
    in_channels = encoder.out_channels

    # first channel is always the image channel -- remove it
    in_channels = in_channels[1:]
    in_channels = in_channels[::-1]

    center = lambda in_c: ASPPCenterBlock(in_c, 256, [1, 2, 3, 4], 512)
    block = lambda in_c, skip_c, out_c, stage: BasicUnetBlock(in_c, skip_c, out_c)

    # [(1024), 512, 256, 64, 64, 32]
    out_channels = [256, 128, 64, 32]
    decoder = UnetDecoder(in_channels, out_channels, block, center)

    head = SegmentationHead(out_channels[-1], out_channels=1, upsampling=2)

    unet = UnetBase(encoder, decoder, head)

    return unet

from .unet_blocks import AttentionGate, AttentionUnetBlock
def AttentionUnetRes50(use_inter_head=False):
    encoder = get_encoder(
        'resnet50', weights='imagenet', in_channels=3
    )
    in_channels = encoder.out_channels

    # first channel is always the image channel -- remove it
    in_channels = in_channels[1:]
    in_channels = in_channels[::-1]

    # compress = 2 / 4
    center = lambda in_c: BasicCenterBlock(in_c, compress=4)
    att = lambda in_c, skip_c: AttentionGate(in_c, skip_c) if skip_c != 0 else nn.Identity()
    block = lambda in_c, skip_c, out_c, stage: AttentionUnetBlock(in_c, skip_c, out_c, att)

    if use_inter_head:
        # out_channels = [512, 256, 128, 32] # 1/2 of skip channels
        out_channels = [256, 128, 64, 32]  # 1/4 of skip channels
        head = SegmentationHead(out_channels[-1], out_channels=1, upsampling=2)
    else:
        in_channels = list(in_channels) + [0]
        out_channels = [256, 128, 64, 32, 16]  # 1/4 of skip channels
        head = SegmentationHead(out_channels[-1], out_channels=1, upsampling=1)

    decoder = UnetDecoder(in_channels, out_channels, block, center)

    unet = UnetBase(encoder, decoder, head)

    return unet

from .unet_center import ViTCenterBlock
from .unet_encoder import ViTUnetEncoder


def ViTUnet(use_inter_head=True):
    encoder = ViTUnetEncoder()

    in_channels = encoder.out_channels
    # first channel is always the image channel -- remove it
    in_channels = in_channels[1:]
    in_channels = in_channels[::-1]

    center = lambda in_c: ViTCenterBlock(in_c)
    block = lambda in_c, skip_c, out_c, stage: BasicUnetBlock(in_c, skip_c, out_c)

    if use_inter_head:
        out_channels = [256, 128, 64]
        head = SegmentationHead(out_channels[-1], out_channels=1, upsampling=2)
    else:
        in_channels = list(in_channels) + [0]
        out_channels = [256, 128, 64, 16]
        head = SegmentationHead(out_channels[-1], out_channels=1, upsampling=1)

    decoder = UnetDecoder(in_channels, out_channels, block, center)
    unet = UnetBase(encoder, decoder, head)

    return unet

from .unet_blocks import TransformerGate
def create_trans_gate(in_c, skip_c):
    if in_c == 512:
        return TransformerGate(in_c, skip_c, 1, 1024, dim=512, depth=2)
    else:
        return TransformerGate(in_c, skip_c, 512//in_c, 1024, depth=2)

def create_trans_gated_block(in_c, skip_c, out_c, stage):
    if stage > 2:
        return BasicUnetBlock(in_c, skip_c, out_c)
    else:
        return AttentionUnetBlock(in_c, skip_c, out_c, create_trans_gate)

class TransCenterBlock(BasicCenterBlock):
    def __init__(self, in_channels):
        super(TransCenterBlock, self).__init__(in_channels)
        self.func = nn.Sequential(
            TransformerGate(in_channels = 0, skip_channels = in_channels, patch_size = 1, patch_num=256),
            self.func
        )
def ViTGateUnet():
    encoder = get_encoder('resnet50', weights='imagenet', depth=4)
    in_channels = encoder.out_channels

    in_channels = in_channels[1:]
    in_channels = in_channels[::-1]

    center = lambda in_c: TransCenterBlock(in_c)
    # no interpolate in segmentation head
    in_channels = list(in_channels) + [0]
    out_channels = [256, 128, 64, 32] # 512, 256, 128, 64
    decoder = UnetDecoder(in_channels, out_channels,
        create_trans_gated_block, center)
    head = SegmentationHead(out_channels[-1], out_channels=1, upsampling=1)

    unet = UnetBase(encoder, decoder, head)

    return unet

if __name__ == '__main__':
    # m = BasicUnet()
    # m = BasicUnetSE50(True, False)
    # m = ShuffleUnetSE50()
    # m = FPNUnetSE50()
    # m = ASPPUnetSE50()
    # m = train_models()
    m = ViTGateUnet()
    print(m)
    x = torch.randn(8, 3, 256, 256)
    out = m(x)
    print(out.shape)
    # torch.save(m.state_dict(), 'munet_se_50_m.pth')

