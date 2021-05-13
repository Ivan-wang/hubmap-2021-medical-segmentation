import glob
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import albumentations as A

from utils import init_model
from dataset import HuBMAPTrainData

from tqdm import tqdm
from catalyst.utils.metrics import dice
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

IMAGE_MEAN = [0.6951843, 0.48494667, 0.65548706]
IMAGE_STD = [0.13569852, 0.24020114, 0.1574288]

from dataset import load_train_data

def run_train_data(ckpt):
    model = init_model(ckpt, 'model_state_dict')
    model.eval()

    _, loader = load_train_data()

    batchSize = 16
    lossFn = nn.BCEWithLogitsLoss()
    metric = lambda out, t: dice(out, t, threshold=0.5)

    lossTotal = 0
    metricTotal = 0
    tileTotal = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            inputs = batch['image']
            tileTotal += inputs.shape[0]
            targets = batch['mask']

            inputs = inputs.cuda()
            inputs.requires_grad_(False)
            targets = targets.cuda()

            out = model(inputs)
            out = out[:, :, 32:-32, 32:-32]
            targets = targets[:, :, 32:-32, 32:-32]
            loss = lossFn(out, targets)
            m = metric(out, targets)

            lossTotal += loss * inputs.shape[0]
            metricTotal += m * inputs.shape[0]
    print(f'Validation Avg Loss {lossTotal / tileTotal:.4f}, \
        Avg Dice {metricTotal / tileTotal:.4f}')

import torch.nn.functional as F
def run_validation(model, th=0.5):
    model.eval()

    _, val_loader = load_train_data()

    scale_factors = [1, 0.5]
    scale_size = [512, 256]

    metric = lambda out, t: dice(out, t, threshold=th)

    thres = np.linspace(0.3, 0.7, 41)
    metricTotal = torch.zeros(len(thres))
    tileTotal = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            inputs = batch[0]
            targets = batch[1]
            tileTotal += inputs.shape[0]

            masks = []
            for scale in scale_factors:
                if scale != 1:
                    data = F.interpolate(inputs.clone(), scale_factor=scale,
                        mode='bilinear', align_corners=False,
                        recompute_scale_factor=False).cuda()
                else:
                    data = inputs.clone().cuda()

                mask = model(data)
                mask = torch.sigmoid(mask).cpu()

                if scale != 1:
                    mask = F.interpolate(mask, scale_factor=1/scale,
                        mode='bilinear', align_corners=False,
                        recompute_scale_factor=False)
                masks.append(mask.clone())
            masks = torch.stack(masks, dim=-1).mean(dim=-1)
            m = torch.zeros(len(thres))
            for i, t in enumerate(thres):
                m[i] = dice(masks[:, :, 32:-32, 32:-32], targets[:, :, 32:-32, 32:-32],
                    threshold=t)

            metricTotal += m * inputs.shape[0]

    avgMetric = metricTotal / tileTotal
    print(f'Validation Avg Dice {avgMetric}')
    print(f'Validation Max Dice {avgMetric.max():.4f}, Best Thres {thres[avgMetric.argmax()]}')

if __name__ == "__main__":
    from models import train_models
    CKPT = 'trained_models/all_mix_scale/all_mix_scale.pth'
    model = train_models()
    model.load_state_dict(torch.load(CKPT))
    model = model.cuda()
    run_validation(model)
