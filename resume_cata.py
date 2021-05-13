import segmentation_models_pytorch as smp
from catalyst.contrib.nn import RAdam, Lookahead
from catalyst.dl import SupervisedRunner
from catalyst.dl import DiceCallback, IouCallback, CriterionCallback, MetricAggregationCallback
from catalyst.contrib.nn import DiceLoss, IoULoss, BCEDiceLoss
from torch import optim
from torch import nn
from catalyst import utils

from dataset import train_transform, HuBMAPTrainData
from tile_dataset import UpdateRandomTileCallback
from utils import CenterDiceCallback

from models import FPNUnetSE50
from models import ASPPUnetSE50

import glob
import numpy as np
from torch.utils.data import DataLoader
def load_fixed_train_tiles(root='./data/selected'):
    ids = glob.glob(root+'/*')
    trainImages, trainMasks = [], []
    for iid in ids:
        images = glob.glob(iid+'/tiles/*.png')
        masks = [p.replace('tiles', 'masks') for p in images]
        trainImages.append(images) # use all selected tiles for finetune
        trainMasks.append(masks)

    trainImages = np.concatenate(trainImages)
    trainMasks = np.concatenate(trainMasks)

    # build data loader
    train_trans = train_transform()
    train_ds = HuBMAPTrainData(trainImages, trainMasks, train_trans)
    trainloader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=16)

    return trainloader

def init_model(CKPT, state_dict='model_state_dict'):
    model = smp.Unet(encoder_name="se_resnet50", encoder_weights=None,
            in_channels=3, classes=1, decoder_attention_type='scse')

    model.load_state_dict(torch.load(CKPT)[state_dict])
    return model

def infer(model, loaders, logdir='logs'):
    pass
def train(model, loaders, logdir='logs/'):
    learning_rate = 1e-4
    # base_optimizer = RAdam(model.parameters(), lr=learning_rate, weight_decay=0.0003)
    # optimizer = Lookahead(base_optimizer)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16)

    num_epochs = 2
    # by default SupervisedRunner uses "features" and "targets",
    # in our case we get "image" and "mask" keys in dataset __getitem__
    device = utils.get_device()
    runner = SupervisedRunner(device=device, input_key="image", input_target_key="label")

    # we have multiple criterions
    criterion = {
        # "dice": DiceLoss(),
        # "iou": IoULoss(),
        # "bce": nn.BCEWithLogitsLoss()
        "bce_dice": BCEDiceLoss(threshold=0.5, bce_weight=1., dice_weight=0.1)
    }

    callbacks = [
        CriterionCallback(
            input_key="label",
            prefix="loss",
            criterion_key="bce_dice"
        ),

        # metrics
        DiceCallback(input_key="label", threshold=0.5),
        IouCallback(input_key="label", threshold=0.5),
        CenterDiceCallback(input_key='label', threshold=0.5),
        # only need this for random tile dataset
        # UpdateRandomTileCallback()
    ]
    # run the validation first
    runner.infer(
        model=model,
        loaders={'valid': loaders['valid']},
        callbacks=callbacks[1:],
        logdir=logdir,
        verbose=True
    )

    # then finetune
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        # our dataloaders
        loaders=loaders,
        # We can specify the callbacks list for the experiment;
        callbacks=callbacks,
        # path to save logs
        logdir=logdir,
        num_epochs=num_epochs,
        # save our best checkpoint by IoU metric
        main_metric="center_dice",
        # IoU needs to be maximized.
        minimize_metric=False,
        # for FP16. It uses the variable from the very first cell
        fp16=None,
        # prints train logs
        verbose=True,
    )

def train_kfold():
    for i, (train_loader, val_loader) in enumerate(load_kfold_data()):
        model = init_model()
        loaders = {'train': train_loader, 'valid': val_loader}
        train(model, loaders, 'kfold/unet_scse_60_fold_{}'.format(i))

import torch
from tile_dataset import load_train_data
def finetune_one_fold(CKPT):
    model = init_model(CKPT)
    # train_loader = load_fixed_train_tiles()
    train_loader, val_loader = load_train_data('./data/zarr_hand', './data/zarr_hand_mask')
    loaders = {'train': train_loader, 'valid': val_loader}
    print('data loaded...')
    train(model, loaders, 'trained_models/fine_tune_hand_label')

if __name__ == '__main__':
    CKPT = 'kfold/unet_scse_60_fold_1/checkpoints/train.52.pth'
    finetune_one_fold(CKPT)
    # train_kfold()

