from tqdm import tqdm
# from rasterio.windows import Window
import segmentation_models_pytorch as smp
import pandas as pd
import numpy as np
import tifffile as tf
# import rasterio
import torch
import cv2
import gc

ROOT = '/home/ivan/workspace/HuBMAP/data/'
TEST = ROOT + 'test/'
TRAIN = ROOT + 'train/'

def init_model():
    model = smp.Unet(encoder_name='se_resnet50',
        encoder_weights=None, in_channels=3, classes=1)
    model = model.cuda()

    ckpt = torch.load('models/bce_adam_cosine_mix.pth')
    model.load_state_dict(ckpt['model_state_dict'])

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

def load_tiff(f):
    tiff = rasterio.open(f, num_threads='all_cpus')
    print(tiff.shape)
    print(type(tiff))


def inference():
    model = init_model()
    model.eval()
    submit_df = pd.read_csv(ROOT+'sample_submission.csv')

    tile_size = 256
    scale = 4
    S_thres = 40
    p_th = 200

    img_mean = np.array([0.6951843, 0.48494667, 0.65548706]).astype(np.float32).reshape((1, 1, 3))
    img_std = np.array([0.13569852, 0.24020114, 0.1574288]).astype(np.float32).reshape((1, 1, 3))

    with torch.no_grad():
        for iid in submit_df.iloc[:, 0]:
            print('Predicting {}... '.format(iid), end=' ')
            tiff = tf.imread('{}/{}.tiff'.format(TEST, iid))
            tiff = np.squeeze(tiff)
            if (tiff.shape[-1] != 3): tiff = tiff.transpose([1, 2, 0])
            rawH, rawW = tiff.shape[:2]
            print(f'TIFF Shape: {rawH} x {rawW}', end=' ')

            size = tile_size * scale
            hpad = (tiff.shape[0] // size + 1) * size - tiff.shape[0]
            wpad = (tiff.shape[1] // size + 1) * size - tiff.shape[1]

            tiff = np.pad(tiff, [[hpad//2, hpad-hpad//2], [wpad//2, wpad-wpad//2], [0, 0]])
            H, W = tiff.shape[:2]
            scaled_tiff = cv2.resize(tiff, (W//scale, H//scale))
            del tiff
            gc.collect()

            scaledH, scaledW = scaled_tiff.shape[:2]
            scaled_tiff = scaled_tiff.reshape((scaledH//tile_size, tile_size, scaledW//tile_size, tile_size, 3)
                ).transpose([0, 2, 1, 3, 4]).reshape((-1, tile_size, tile_size, 3))
            print('Scaled tiff shape: ', scaled_tiff.shape)

            mask = np.zeros((scaled_tiff.shape[0], size, size), dtype=np.uint8)
            for idx in tqdm(range(scaled_tiff.shape[0])):
                imgHSV = cv2.cvtColor(scaled_tiff[idx], cv2.COLOR_RGB2HSV)
                if (imgHSV[:, :, 1]>S_thres).sum() <= p_th or scaled_tiff[idx].sum() <= p_th: continue
                else :
                    img = ((scaled_tiff[idx].astype(np.float32) / 255.) - img_mean) / img_std
                    timg = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
                    out = torch.sigmoid(model(timg))
                    out = torch.nn.functional.upsample_bilinear(out, scale_factor=scale)
                    out = (out > 0.5).long().cpu().numpy()
                    mask[idx] = np.squeeze(out)
            del scaled_tiff
            gc.collect()

            mask = mask.reshape((scaledH//tile_size, scaledW//tile_size, size, size)
                ).transpose(0, 2, 1, 3).reshape((H, W))
            # mask = cv2.resize(mask, (scaledW*scale, scaledH*scale), interpolation=cv2.INTER_LINEAR)
            mask = mask[hpad//2:-(hpad-hpad//2), wpad//2:-(wpad-wpad//2)]
            mask = np.where(mask > 0.5, 1, 0).astype(np.uint8)
            print('Predicted mask shape: {}'.format(mask.shape))
            submit_df.loc[submit_df.id == iid, 'predicted'] = rle_encode_less_memory(mask)
            del mask
            gc.collect()
    # with torch.no_grad():
    #     for iid in submit_df.iloc[:, 0]:
    #         print('Predicting {}... '.format(iid), end=' ')
    #         identity = rasterio.Affine(1, 0, 0, 0, 1, 0)
    #         data = rasterio.open('{}/{}.tiff'.format(TEST, iid),
    #                              transform=identity, num_threads='all_cpus')
    #         rawH, rawW = data.shape[:2]
    #         print(f'TIFF Shape: {rawH} x {rawW}', end=' ')

    #         size = tile_size * scale
    #         hpad = (rawH // size + 1) * size - rawH
    #         wpad = (rawW // size + 1) * size - rawW
    #         winSizeH = (rawH + hpad) // size
    #         winSizeW = (rawW + wpad) // size

    #         mask = np.zeros((winSizeH*winSizeW, size, size), dtype=np.uint8)
    #         for idx in tqdm(range(winSizeH*winSizeW)):
    #             t0, t1 = idx // winSizeW, idx % winSizeW
    #             x0, y0 = -hpad // 2 + t0 * size, -wpad // 2 + t1 * size
    #             t00, t01 = max(0, x0), min(x0+size, rawH)
    #             t10, t11 = max(0, y0), min(y0+size, rawW)

    #             tile = np.zeros((size, size, 3), np.uint8)
    #             tile[(t00-x0):(t01-x0), (t10-y0):(t11-y0)] = data.read([1, 2, 3],
    #                window=Window.from_slices((t00, t01), (t10, t11))).transpose([1, 2, 0])
    #             tile = cv2.resize(tile, (tile_size, tile_size))
    #             imgHSV = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
    #             if (imgHSV[:, :, 1]>S_thres).sum() <= p_th or tile.sum() <= p_th: continue
    #             else :
    #                 img = ((tile.astype(np.float32) / 255.) - img_mean) / img_std
    #                 timg = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
    #                 out = torch.sigmoid(model(timg))
    #                 out = torch.nn.functional.upsample_bilinear(out, scale_factor=scale)
    #                 out = (out > 0.5).long().cpu().numpy()
    #                 mask[idx] = np.squeeze(out)
    #         del data
    #         gc.collect()

    #         mask = mask.reshape((winSizeH, winSizeW, size, size)
    #             ).transpose(0, 2, 1, 3).reshape((winSizeH*size, winSizeW*size))
    #         # mask = cv2.resize(mask, (scaledW*scale, scaledH*scale), interpolation=cv2.INTER_LINEAR)
    #         mask = mask[hpad//2:-(hpad-hpad//2), wpad//2:-(wpad-wpad//2)]
    #         # mask = np.where(mask > 0.5, 1, 0).astype(np.uint8)
    #         print('Predicted mask shape: {}'.format(mask.shape))
    #         submit_df.loc[submit_df.id == iid, 'predicted'] = rle_encode_less_memory(mask)
    #         del mask
    #         gc.collect()

    submit_df.to_csv('submission.csv', index=False)
if __name__ == '__main__':
    inference()
    # submit_df = pd.read_csv(ROOT+'sample_submission.csv')
    # load_tiff('{}/{}.tiff'.format(TEST, submit_df.iloc[0, 0]))%
