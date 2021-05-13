from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch import ToTensorV2
import albumentations as A
import tifffile as tf
import pandas as pd
import numpy as np
import torch
import cv2
import glob
import os

ROOT = '/home/ivan/workspace/HuBMAP/raw_data/'
TRAIN = ROOT+'train/'
TEST = ROOT + 'test/'
TRAINDF = ROOT + 'train.csv'

class HuBMAPTrainData(Dataset):
    def __init__(self, tiles, masks, transform=None):
        super(HuBMAPTrainData, self).__init__()
        self.dataPath = tiles
        self.maskPath = masks
        self.transform = transform
        self.tiles = None
        self.masks = None
        self.load_images()

    def load_images(self):
        self.tiles = [cv2.cvtColor(cv2.imread(t), cv2.COLOR_BGR2RGB) for t in self.dataPath]
        self.masks = [cv2.imread(m).astype(np.float32)[:, :, 0:1] / 255. for m in self.maskPath]

    def __len__(self):
        return len(self.dataPath)

    def __getitem__(self, idx):
        if self.tiles is not None:
            tile = self.tiles[idx]
        else:
            tile = self.dataPath[idx]
            tile = cv2.imread(tile)
            tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)# .astype(np.float32)
            # tile /= 255.

        if self.masks is not None:
            mask = self.masks[idx]
        else:
            mask = self.maskPath[idx]
            mask = cv2.imread(mask).astype(np.float32)
            mask = mask[:, :, 0:1] / 255.
        # mask = mask.transpose([2, 0, 1])

        if self.transform is not None:
            transformed = self.transform(image=tile, mask=mask)
            tile = transformed['image']
            mask = transformed['mask']

        return {'image': tile, 'label': mask}
        # return tile, mask

class HuBMAPData(Dataset):
    def __init__(self, iid, rle, scale=None, tile_size=256, transform=None):
        super(HuBMAPData, self).__init__()

        self.iid = iid
        self.tile_size = tile_size

        tiff = tf.imread('{}/{}.tiff'.format(TRAIN, self.iid))
        if (tiff.shape[-1] != 3): tiff = tiff.transpose([1, 2, 0])
        mask = self._rle2mask(rle, tiff)
        tiff, mask = self._scale(tiff, mask, scale)
        hpad = (tiff.shape[0] // self.tile_size + 1) * self.tile_size - tiff.shape[0]
        wpad = (tiff.shape[1] // self.tile_size + 1) * self.tile_size - tiff.shape[1]
        self.tiff = np.pad(tiff, [[hpad//2, hpad-hpad//2], [wpad//2, wpad-wpad//2], [0, 0]])
        self.mask = np.pad(mask, [[hpad//2, hpad-hpad//2], [wpad//2, wpad-wpad//2]])

        self.tile_row = self.tiff.shape[0] // self.tile_size
        self.tile_col = self.tiff.shape[1] // self.tile_size

        self.transform = transform

    def _scale(self, tiff, mask, scale=None):
        if scale is None:
            return tiff, mask
        H, W = tiff.shape[:2]
        scaled_tiff = cv2.resize(tiff, (H//scale, W//scale))
        scaled_mask = cv2.resize(mask, (H//scale, W//scale), interpolation=cv2.INTER_NEAREST)

        return scaled_tiff, scaled_mask

    def _rle2mask(self, rle, tiff):
        s = rle.split()
        starts = np.array(s[0::2], dtype=np.int)
        lengths = np.array(s[1::2], dtype=np.int)
        starts -= 1
        ends = starts + lengths
        mask = np.zeros(tiff.shape[0] * tiff.shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            mask[lo : hi] = 1
        return mask.reshape((tiff.shape[1], tiff.shape[0])).T

    def __len__(self):
        return self.tile_row * self.tile_col

    def __getitem__(self, idx):
        row = idx // self.tile_col
        col = idx % self.tile_col

        img = self.tiff[row*self.tile_size:(row+1)*self.tile_size, col*self.tile_size:(col+1)*self.tile_size, :]
        mask = self.mask[row*self.tile_size:(row+1)*self.tile_size, col*self.tile_size:(col+1)*self.tile_size]

        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img, mask = transformed['image'], transformed['mask']

        return img, mask

from sklearn.model_selection import train_test_split
def train_transform():
    trans = A.Compose([
        A.OneOf([
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ], p=0.5),
        A.RandomRotate90(),
        # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9,
                         # border_mode=cv2.BORDER_REFLECT),
        # Shape
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.3),
        # Color
        A.OneOf([
            A.HueSaturationValue(10,15,10),
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.Normalize(mean=[0.6951843, 0.48494667, 0.65548706], std=[0.13569852, 0.24020114, 0.1574288], max_pixel_value=255, always_apply=True),
        ToTensorV2(transpose_mask=True, always_apply=True)
    ])

    return trans

def build_dataset(trainImages, trainMasks, valImages, valMasks):
    train_trans = train_transform()
    train_ds = HuBMAPTrainData(trainImages, trainMasks, train_trans)

    val_trans = A.Compose([
        A.Normalize(mean=[0.6951843, 0.48494667, 0.65548706], std=[0.13569852, 0.24020114, 0.1574288],
            max_pixel_value=255, always_apply=True),
        ToTensorV2(transpose_mask=True, always_apply=True)
    ])
    val_ds = HuBMAPTrainData(valImages, valMasks, val_trans)

    return train_ds, val_ds

def val_transform():
    trans = A.Compose([
        A.Normalize(mean=[0.6951843, 0.48494667, 0.65548706], std=[0.13569852, 0.24020114, 0.1574288],
            max_pixel_value=255, always_apply=True),
        ToTensorV2(transpose_mask=True, always_apply=True)
    ])
    return trans

import torch.nn.functional as F
def train_collate_fn(batch):
    tiles = torch.stack([b[0] for b in batch], dim=0)
    masks = torch.stack([b[1] for b in batch], dim=0)
    if (torch.rand(1) > 0.5).any():
        tiles = F.interpolate(tiles, scale_factor=0.5, mode='bilinear', align_corners=False,
            recompute_scale_factor=False)
        masks = F.interpolate(masks, scale_factor=0.5, mode='nearest',
            recompute_scale_factor=False)
    return tiles, masks

def build_data_loader(trainImages, trainMasks, valImages, valMasks):
    train_ds, val_ds = build_dataset(trainImages, trainMasks, valImages, valMasks)

    trainloader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=16)
            # collate_fn=train_collate_fn)
    valloader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=16)

    return trainloader, valloader

def gather_train_data_path(root='./data/datax4_512'):
    ids = glob.glob(root+'/*')
    trainImages, trainMasks = [], []
    valImages, valMasks = [], []
    for iid in ids:
        images = glob.glob(iid+'/tiles/*.png')
        masks = [p.replace('tiles', 'masks') for p in images]
        ti, vi, tm, vm = train_test_split(images, masks, test_size=0.2, random_state=17)
        trainImages.append(ti)
        valImages.append(vi)
        trainMasks.append(tm)
        valMasks.append(vm)
    trainImages = np.concatenate(trainImages)
    valImages = np.concatenate(valImages)
    trainMasks = np.concatenate(trainMasks)
    valMasks = np.concatenate(valMasks)

    return trainImages, trainMasks, valImages, valMasks

def load_train_data(root='./data/datax4_512'):
    trainImages, trainMasks, valImages, valMasks = gather_train_data_path()
    return build_data_loader(trainImages, trainMasks, valImages, valMasks)

def load_train_dataset():
    trainImages, trainMasks, valImages, valMasks = gather_train_data_path()
    return build_dataset(trainImages, trainMasks, valImages, valMasks)


from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
def load_kfold_data():
    trainRoot = './data/datax4_512'
    ids = glob.glob(trainRoot+'/*')

    images = []
    for iid in ids: images.extend(glob.glob(iid+'/tiles/*.png'))
    masks = [p.replace('tiles', 'masks') for p in images]
    images = np.asarray(images)
    masks = np.asarray(masks)

    kf = KFold(n_splits=5, shuffle=True, random_state=17)
    for t, v in kf.split(images, masks):
        yield build_data_loader(images[t], masks[t], images[v], masks[v])

def load_kfold_dataset():
    trainRoot = './data/datax4_512'
    ids = glob.glob(trainRoot+'/*')

    images = []
    iid_labels = []
    for il, iid in enumerate(ids):
        iid_image = glob.glob(iid+'/tiles/*.png')
        images.extend(iid_image)
        iid_labels.extend([il]*len(iid_image))
    masks = [p.replace('tiles', 'masks') for p in images]
    images = np.asarray(images)
    masks = np.asarray(masks)
    iid_labels = np.asarray(iid_labels)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
    for t, v in kf.split(images, iid_labels):
        yield build_dataset(images[t], masks[t], images[v], masks[v])

if __name__ == '__main__':
    loader, val = load_train_data()
    for batch in loader:
        print(batch['image'].min(), batch['image'].max())
        break
    for batch in val:
        print(batch['image'].min(), batch['image'].max())
        break
