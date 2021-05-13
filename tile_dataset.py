from torch.utils.data import Dataset, DataLoader
import numpy as np
import tifffile as tf
import cv2
import glob
import zarr
import os
import gc
from tqdm import tqdm

from dataset import train_transform, val_transform

# refer to deepflash2
class DeformationField:
    "Creates a deformation field for data augmentation"
    def __init__(self, shape=(540, 540), scale=1):
        self.shape, self.scale = shape, scale
        grid_range = [np.linspace(-(d*self.scale)/2, ((d*self.scale)/2)-1, d) for d in shape]
        self.deformationField = np.meshgrid(*grid_range)[::-1]
        self.orders = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC]

    def get(self, offset=(0, 0), pad=(0, 0)):
        "Get relevant slice from deformation field"
        sliceDef = tuple(slice(int(p / 2), int(-p / 2)) if p > 0 else None for p in pad)
        deform = [d[sliceDef] for d in self.deformationField]
        return [d + offs for (d, offs) in zip(deform, offset)]

    def apply(self, data, offset=(0, 0), pad=(0, 0), order=1):
        "Apply deformation field to image using interpolation"
        outshape = tuple(int(s - p) for (s, p) in zip(self.shape, pad))
        coords = [np.squeeze(d).astype('float32').reshape(*outshape) for d in self.get(offset, pad)]
        # Get slices to avoid loading all data (.zarr files)
        sl = []
        for i in range(len(coords)):
            cmin, cmax = int(coords[i].min()), int(coords[i].max())
            dmax = data.shape[i]
            if cmin<0:
                cmax = max(-cmin, cmax)
                cmin = 0
            elif cmax>dmax:
                cmin = min(cmin, 2*dmax-cmax)
                cmax = dmax
                coords[i] -= cmin
            else: coords[i] -= cmin
            sl.append(slice(cmin, cmax))
        if len(data.shape) == len(self.shape) + 1:
            tile = np.empty((*outshape, data.shape[-1]))
            for c in range(data.shape[-1]):
                tile[..., c] = cv2.remap(data[sl[0],sl[1], c], coords[1],coords[0], interpolation=order, borderMode=cv2.BORDER_REFLECT)
        else:
            tile = cv2.remap(data[sl[0], sl[1]], coords[1], coords[0], interpolation=order, borderMode=cv2.BORDER_REFLECT)
        return tile

class TileDataset(Dataset):
    "Pytorch Dataset that creates random tiles for validation and prediction on new data."
    n_inp = 1
    def __init__(self, data_root, tile_shape=(512, 512), padding=(0, 0), scale=2, label_root=None, is_zarr=False, val_length=None, val_seed=42, transform=None):
        super(TileDataset, self).__init__()
        self.tile_shape = tile_shape
        self.padding = padding
        self.scale = scale

        self.output_shape = tuple(int(t - p) for (t, p) in zip(self.tile_shape, self.padding))
        self.image_indices = []
        self.image_shapes = []
        self.in_slices = []
        self.out_slices = []
        self.centers = []
        self.valid_indices = None
        self.transform = transform
        # NOTE: scaling is done be the tiler
        self.tiler = DeformationField(self.tile_shape, scale=scale)

        if is_zarr:
            self.data = zarr.open(data_root, mode='r')
            self.files = list(self.data.keys())
            is_zarr = True
        else:
            self.files = glob.glob(f'{data_root}/*.tiff')
            root = zarr.group(store=zarr.storage.TempStore(), overwrite=True)
            self.data = root.create_group('data')
            is_zarr = False

        j = 0
        for i, k in enumerate(tqdm(self.files, leave=False)):
            if is_zarr:
                img = self.data[k]
            else:
                img = self._read_img(k, divide=255.)
                name = os.path.basename(k).split('.')[0]
                self.data[name] = img
            # Tiling
            data_shape = tuple(int(x//self.scale) for x in img.shape[:-1])
            for ty in range(max(1, int(np.ceil(data_shape[0] / self.output_shape[0])))):
                for tx in range(max(1, int(np.ceil(data_shape[1] / self.output_shape[1])))):
                    self.centers.append((int((ty + 0.5) * self.output_shape[0]*self.scale),
                                        int((tx + 0.5) * self.output_shape[1]*self.scale)))
                    self.image_indices.append(i)
                    self.image_shapes.append(data_shape)
                    sliceDef = tuple(slice(int(tIdx * o), int(min((tIdx + 1) * o, s))) for (tIdx, o, s) in zip((ty, tx), self.output_shape, data_shape))
                    self.out_slices.append(sliceDef)
                    sliceDef = tuple(slice(0, int(min((tIdx + 1) * o, s) - tIdx * o)) for (tIdx, o, s) in zip((ty, tx), self.output_shape, data_shape))
                    self.in_slices.append(sliceDef)
                    j += 1
        del img
        gc.collect()

        # convert to path to zarr keys
        if not is_zarr:
            self.files = [os.path.basename(k).split('.')[0] for k in self.files]

        if label_root is not None:
            assert is_zarr
            self.labels = zarr.open(label_root, mode='r')
        else:
            self.labels = None

        if val_length:
            if val_length>len(self.image_shapes):
                print(f'Reducing validation from lenght {val_length} to {len(self.image_shapes)}')
                val_length = len(self.image_shapes)
            np.random.seed(val_seed)
            choice = np.random.choice(len(self.image_indices), val_length, replace=False)
            self.valid_indices = {i:idx for i, idx in  enumerate(choice)}

    def _read_img(self, path):
        "Read image and normalize to 0-1 range"
        if os.path.basename(path).split('.')[-1] == '.zarr':
            img = zarr.convenience.open(path.as_posix())
            if len(img.shape)==4: # assuming shape (z_dim, n_channel, y_dim, x_dim)
                img = np.max(img, axis=0) # max z projection
                img = np.moveaxis(img, 0, -1)
        else:
            img = tf.imread(path)
            if len(img.shape) == 5:
                img = img.squeeze().transpose(1, 2, 0)
            elif img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        return img

    def __len__(self):
        if self.valid_indices: return len(self.valid_indices)
        else: return len(self.image_shapes)

    def __getitem__(self, idx):
        if self.valid_indices: idx = self.valid_indices[idx]
        k = self.files[self.image_indices[idx]]
        img = self.data[k]
        centerPos = self.centers[idx]
        # NOTE: scaling is done by the tiler
        X = self.tiler.apply(img, centerPos)
        X = X.astype('float32')
        if self.labels is not None:
            lbl = self.labels[k]
            Y = self.tiler.apply(lbl, centerPos, self.padding, order=0).astype('float32')
        else:
            Y = None

        if self.transform is not None:
            transformed = self.transform(image=X, mask=Y)
            X = transformed['image']
            Y = transformed['mask']

        return {'image': X, 'mask': Y, 'image_id': k}

    def reconstruct_from_tiles(self, tiles):
        "Reconstruct masks or predictions from list of tiles"

        assert isinstance(tiles, list), "You need to pass a list"
        assert len(tiles) == len(self), f"Tile list must have length{len(self)}"

        out_ll = []
        for idx in range(len(self)):
            outIdx = self.image_indices[idx]
            outShape = self.image_shapes[idx]
            outSlice = self.out_slices[idx]
            inSlice = self.in_slices[idx]
            if len(out_ll) < outIdx + 1:
                if len(tiles[0].shape)>2:
                    out_ll.append(np.empty((*outShape, self.c)))
                else:
                    out_ll.append(np.empty(outShape))
            out_ll[outIdx][outSlice] = tiles[idx][inSlice]

        return out_ll


import matplotlib.pyplot as plt
def draw_batch(images, masks):
    fig, ax = plt.subplots(4, 4, figsize=(20, 20))

    for i, ax in enumerate(ax.ravel()):
        ax.imshow(images[i], alpha=0.8)
        ax.imshow(masks[i], alpha=0.2)

    plt.savefig('./visual_results/batch_tile.png')
    plt.close()

def test_tile_dataset():
    data_root = './data/zarr_data'
    label_root = './data/zarr_mask/label'
    # to visulize the batch, do not use normalization in transform
    ds = TileDataset(data_root, label_root=label_root, is_zarr=True, transform=val_transform())
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    for data in loader:
        print(type(data['image']), data['image'].shape, data['image'].dtype,
            data['image'].min(), data['image'].max())
        print(type(data['mask']), data['mask'].shape, data['mask'].dtype,
            data['mask'].min(), data['mask'].max())
        break
    images = data['image'].permute(0, 2, 3, 1).numpy() / 255.
    labels = data['mask'].numpy()
    draw_batch(images, labels)

tile_config = {
    'tile_shape': (512, 512),
    'padding': (0, 0),
    'scale': 2 # = 2 x 2 = 4
}
deformation_config = {
    'grid': (150, 150),
    'magnitude': (10, 10),
    'p_zoom': 0.75, # probability to change scale
    'zoom_sigma': 0.1
}
class RandomTileDataset(Dataset):
    """
    Pytorch Dataset that creates random tiles with augmentations from the input images.
    """
    n_inp = 1
    def __init__(self, data_root, label_root, random_scale=True, sample_mult=None, tile_cfg=tile_config, deformation_cfg=deformation_config, transform=None):
        super(RandomTileDataset, self).__init__()
        self.data_root = data_root
        self.tile_cfg = tile_config
        self.sample_mult = sample_mult
        self.deformation_cfg = deformation_cfg
        self.transform = transform


        # load data, label, etc.
        self.data = zarr.open(data_root, 'r')
        self.files = list(self.data.keys())
        print(self.files)

        self.tiler = None
        self.random_scale = random_scale
        self.init_tiler()

        labels = zarr.open(label_root)
        self.labels = labels['label']
        # need to prepare regions
        self.pdfs = labels['pdfs']

        # Sample mulutiplier: Number of random samplings from augmented image
        if self.sample_mult is None:
            tile_shape = np.array(self.tile_cfg['tile_shape'])-np.array(self.tile_cfg['padding'])
            msk_shape = np.array(self.labels[self.files[0]].shape[:-1])
            self.sample_mult = int(np.product(np.floor(msk_shape/tile_shape)))

    def __len__(self):
        return len(self.files)*self.sample_mult

    def _random_center(self, pdf, orig_shape, scale=512):
        # NOTE: scale parameter should match the preprocessing
        scale_y = int((orig_shape[1]/orig_shape[0])*scale)
        cx, cy = np.unravel_index(np.argmax(pdf > np.random.random()), (scale,scale_y))
        cx = int(cx*orig_shape[0]/scale)
        cy = int(cy*orig_shape[1]/scale_y)
        return cx, cy

    def __getitem__(self, idx):
        idx = idx % len(self.files)

        k = self.files[idx]
        img = self.data[k]
        n_channels = img.shape[-1]

        lbl, pdf  = self.labels[k], self.pdfs[k]

        center = self._random_center(pdf[:], lbl.shape)
        X = self.tiler.apply(img, center).astype('uint8') # no normalization
        Y = self.tiler.apply(lbl, center, self.tile_cfg['padding'], 0).astype('float32')

        if self.transform is not None:
            transformed = self.transform(image=X, mask=Y)
            X = transformed['image'].float()
            Y = transformed['mask'].unsqueeze(0)

        return {'image': X, 'label': Y}

    def init_tiler(self):
        # augmentation is done by transformer
        self.tile_shape = self.tile_cfg['tile_shape']
        self.scale = self.tile_cfg['scale']

        if self.random_scale and np.random.random()<self.deformation_cfg['p_zoom']:
            self.scale *= np.random.normal(1, self.deformation_cfg['zoom_sigma'])

        self.tiler = DeformationField(self.tile_shape, self.scale)

def test_random_tiles():
    data_root = './data/zarr_data'
    label_root = './data/zarr_mask'
    # to visulize the batch, do not use normalization in transform
    ds = RandomTileDataset(data_root, label_root, transform=train_transform())
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    for batch in loader:
        print(batch['image'].shape, batch['image'].min(), batch['image'].max())
        print(batch['label'].shape, batch['label'].min(), batch['label'].max())
        break

    images = batch['image'].permute(0, 2, 3, 1).numpy() / 255.
    labels = batch['label'].numpy()
    draw_batch(images, labels)

def load_train_data(data_root='./data/zarr_data', label_root='./data/zarr_mask'):
    t_config = tile_config
    # t_config['tile_shape'] = (256, 256)
    # t_config['scale'] = 4
    train_ds = RandomTileDataset(data_root, label_root,
        sample_mult=150, tile_cfg=t_config, transform=train_transform())
    val_ds = RandomTileDataset(data_root, label_root, random_scale=False,
        sample_mult=35, tile_cfg=t_config, transform=val_transform())
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=True, num_workers=16)

    return train_loader, val_loader

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder, CallbackScope
class UpdateRandomTileCallback(Callback):
    def __init__(self):
        super().__init__(order=CallbackOrder.External, node=CallbackNode.all, scope=CallbackScope.stage)

    def on_loader_end(self, runner):
        train_loader = runner.loaders['train']
        train_dataset = train_loader.dataset
        if isinstance(train_dataset, RandomTileDataset):
            train_dataset.init_tiler()

if __name__ == '__main__':
    # test_tiler()
    # test_tile_dataset()
    # test_random_tiles()
    train, val = load_train_data()
    for batch in train:
        print(batch['image'].shape)
        print(batch['label'].shape)
        break

    for batch in val:
        print(batch['image'].shape)
        print(batch['label'].shape)
        break

