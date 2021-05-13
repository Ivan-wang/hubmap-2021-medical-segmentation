from tqdm import tqdm

import pandas as pd
import numpy as np
import tifffile as tf
import cv2
import os

ROOT = '/home/ivan/workspace/HuBMAP/raw_data/'
TRAIN = ROOT+'train/'
TRAINDF = ROOT + 'train.csv'

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
def scale_tiff(tiff, mask, scale=None):
    if scale is None:
        return tiff, mask
    H, W = tiff.shape[:2]
    scaled_tiff = cv2.resize(tiff, (W//scale, H//scale))
    scaled_mask = cv2.resize(mask, (W//scale, H//scale), interpolation=cv2.INTER_NEAREST)
    return scaled_tiff, scaled_mask

def data_preprocess(iid, rle, root, scale=1, tile_size=256):
    tiff = tf.imread('{}/{}.tiff'.format(TRAIN, iid))
    tiff = np.squeeze(tiff)
    if (tiff.shape[-1] != 3): tiff = tiff.transpose([1, 2, 0])
    mask = rle2mask(rle, tiff)
    size = scale * tile_size

    hpad = (tiff.shape[0] // size + 1) * size - tiff.shape[0]
    wpad = (tiff.shape[1] // size + 1) * size - tiff.shape[1]
    tiff = np.pad(tiff, [[hpad//2, hpad-hpad//2], [wpad//2, wpad-wpad//2], [0, 0]])
    mask = np.pad(mask, [[hpad//2, hpad-hpad//2], [wpad//2, wpad-wpad//2]])

    tiff, mask = scale_tiff(tiff, mask, scale)


    tile_row = tiff.shape[0] // tile_size
    tile_col = tiff.shape[1] // tile_size
    print('Image ID: {}, Tiles {}x{}'.format(iid, tile_row, tile_col))

    S_thres = 40
    p_th = 200

    tiff = tiff.reshape((tile_row, tile_size, tile_col, tile_size, 3)).transpose([0, 2, 1, 3, 4]).reshape((-1, tile_size, tile_size, 3))
    mask = mask.reshape((tile_row, tile_size, tile_col, tile_size)).transpose([0, 2, 1, 3]).reshape((-1, tile_size, tile_size))

    for i in tqdm(range(tiff.shape[0])):
        imgHSV = cv2.cvtColor(tiff[i], cv2.COLOR_RGB2HSV)
        if (imgHSV[:, :, 1]>S_thres).sum() <= p_th or tiff[i].sum() <= p_th: continue

        img = cv2.cvtColor(tiff[i], cv2.COLOR_RGB2BGR)
        cv2.imwrite('{}/{}/tiles/{}_{}.png'.format(root, iid, iid, i), tiff[i])
        cv2.imwrite('{}/{}/masks/{}_{}.png'.format(root, iid, iid, i), mask[i])

def crop_with_overlap(iid, rle, root, scale=4, tileSize=320, stepSize=256):
    tiff = tf.imread('{}/{}.tiff'.format(TRAIN, iid))
    if len(tiff.shape) > 3: tiff = np.squeeze(tiff)
    if (tiff.shape[-1] != 3): tiff = tiff.transpose([1, 2, 0])

    rawH, rawW = tiff.shape[:2]
    print(f'Raw TIFF Size: {(rawH, rawW)}')
    mask = rle2mask(rle, tiff)

    rawTileSize = tileSize * scale
    rawStepSize = stepSize * scale

    hpad = (rawH // rawStepSize + 1) * rawStepSize - rawH
    wpad = (rawW // rawStepSize + 1) * rawStepSize - rawW
    nWinH = (rawH + hpad) // rawStepSize
    nWinW = (rawW + wpad) // rawStepSize
    print(f'Padded TIFF Size: {(rawH+hpad, rawW+wpad)}')
    print(f'NWindow: {nWinH}x{nWinW}')

    margin = (rawTileSize-rawStepSize) // 2

    S_thres = 40
    p_th = 200

    for idx in tqdm(range(nWinH*nWinW)):
        row, col = idx // nWinW, idx % nWinW
        H, W = rawStepSize * row, rawStepSize * col
        _H, _W = H - margin, W - margin
        H0, W0 = _H - hpad // 2, _W - wpad // 2
        t00, t01 = max(0, H0), min(H0+rawTileSize, rawH)
        t10, t11 = max(0, W0), min(W0+rawTileSize, rawW)

        tile = np.zeros((rawTileSize, rawTileSize, 3), np.uint8)
        _mask = np.zeros((rawTileSize, rawTileSize), np.uint8)
        tile[(t00-H0):(t01-H0), (t10-W0):(t11-W0)] = tiff[t00:t01, t10:t11]
        _mask[(t00-H0):(t01-H0), (t10-W0):(t11-W0)] = mask[t00:t01, t10:t11]

        scaledTile = cv2.resize(tile, (tileSize, tileSize))
        tileHSV = cv2.cvtColor(scaledTile, cv2.COLOR_RGB2HSV)
        if (tileHSV[:, :, 1]>S_thres).sum() <= p_th or scaledTile.sum() <= p_th: continue
        else:
            scaledMask = cv2.resize(_mask, (tileSize, tileSize),
                interpolation=cv2.INTER_NEAREST)

        img = cv2.cvtColor(scaledTile, cv2.COLOR_RGB2BGR)
        cv2.imwrite('{}/{}/tiles/{}_{}.png'.format(root, iid, iid, idx), scaledTile)
        cv2.imwrite('{}/{}/masks/{}_{}.png'.format(root, iid, iid, idx), scaledMask)

def main():
    traindf = pd.read_csv(TRAINDF)
    print(traindf.head())
    SUB_ROOT = 'datax4_256_240'
    for i in range(traindf.shape[0]):
        iid = traindf.iloc[i, 0]
        if not os.path.isdir('{}/{}'.format(SUB_ROOT, iid)):
            os.mkdir('{}/{}'.format(SUB_ROOT, iid))
            os.mkdir('{}/{}/tiles'.format(SUB_ROOT, iid))
            os.mkdir('{}/{}/masks'.format(SUB_ROOT, iid))
        crop_with_overlap(iid, traindf.iloc[i, 1], SUB_ROOT, 4, 256, 240)


if __name__ == '__main__':
    main()
