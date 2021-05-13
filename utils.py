import segmentation_models_pytorch as smp
import numpy as np
import torch
import cv2

def rle2mask(rle, tiff):
    s = rle.split()
    starts = np.array(s[0::2], dtype=np.int)
    lengths = np.array(s[1::2], dtype=np.int)
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(tiff.shape[0] * tiff.shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo : hi] = 255
    return mask.reshape((tiff.shape[1], tiff.shape[0])).T

def init_model(ckpt, state_dict='state_dict'):
    model = smp.Unet(encoder_name='se_resnet50',
        encoder_weights=None, in_channels=3, classes=1)
    model = model.cuda()

    ckpt = torch.load(ckpt)
    model.load_state_dict(ckpt[state_dict])
    return model

#https://www.kaggle.com/bguberfain/memory-aware-rle-encoding
#with bug fix
def rle_encode_less_memory(img):
    #watch out for the bug
    pixels = img.T.flatten()

    # This simplified method requires first and last pixel to be zero
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)

from catalyst.callbacks.metric import BatchMetricCallback
from catalyst.metrics.dice import dice
from catalyst.metrics.functional import (
    wrap_class_metric2dict,
    wrap_metric_fn_with_activation
)

def center_dice(m):
    metric_fn = lambda outputs, targets, class_dim=1, threshold=None, eps=1e-7: \
        dice(outputs[:, :, m:-m, m:-m], targets[:, :, m:-m, m:-m], class_dim, threshold, eps)
    return metric_fn

class CenterDiceCallback(BatchMetricCallback):
    def __init__(self, tile=512, center=448, input_key='targets', output_key='logits',
            prefix='center_dice', activation='Sigmoid', per_class=False, class_args=None, **kwargs):
        metric_fn = wrap_metric_fn_with_activation(
            metric_fn=center_dice((tile-center)//2),
            activation=activation
        )
        metric_fn = wrap_class_metric2dict(
            metric_fn, per_class=per_class, class_args=class_args)
        super().__init__(prefix=prefix, metric_fn=metric_fn, input_key=input_key,
            output_key=output_key, **kwargs)

