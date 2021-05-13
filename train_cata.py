import segmentation_models_pytorch as smp
from catalyst.contrib.nn import RAdam, Lookahead
from catalyst.dl import SupervisedRunner
from catalyst.dl import DiceCallback, IouCallback, CriterionCallback, MetricAggregationCallback
from catalyst.contrib.nn import DiceLoss, IoULoss, BCEDiceLoss
from torch import optim
from torch import nn
from catalyst import utils

from dataset import load_train_data as load_fixed_tiles
# from dataset import load_kfold_dataset
from dataset import load_kfold_data

# from models import FPNUnetSE50
# from models import ASPPUnetSE50
# from models import BasicUnet
from models import ViTGateUnet
from models import ViTUnet

from tile_dataset import load_train_data
from tile_dataset import UpdateRandomTileCallback
from utils import CenterDiceCallback

def init_model():
    # model = BasicUnet(3)
    # model = ViTGateUnet() # use interpolate in head
    model = ViTUnet()
    # model = smp.Unet(encoder_name="se_resnet50", in_channels=3, classes=1, decoder_attention_type='scse')
    # model = FPNUnetSE50()
    # model = ASPPUnetSE50()
    return model

def train(model, loaders, logdir='logs/'):
    learning_rate = 1e-4
    # base_optimizer = RAdam(model.parameters(), lr=learning_rate, weight_decay=0.0003)
    # optimizer = Lookahead(base_optimizer)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=64)

    num_epochs = 64
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
        UpdateRandomTileCallback()
    ]

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

def train_kfold(num_fold=5):
    # for i, (train_loader, val_loader) in enumerate(load_kfold_data()):
    for i in range(num_fold):
        # NOTE: when using the random tiles, no need to split dataset
        train_loader, val_loader = load_train_data()
        # val_loader, _ = load_fixed_tiles()

        model = init_model()
        loaders = {'train': train_loader, 'valid': val_loader}
        train(model, loaders, 'kfold_hand_label_vit/hand_label_vit_{}'.format(i))

import torch
def train_one_fold():
    model = init_model()
    # NOTE: use random tiles for training
    train_loader, _ = load_train_data()
    _, val_loader = load_fixed_tiles()
    loaders = {'train': train_loader, 'valid': val_loader}
    print('data loaded...')
    train(model, loaders, 'trained_models/unet_se50_scse_mix_data')

if __name__ == '__main__':
    # train_one_fold()
    train_kfold(2)
    # m = ViTUnet()
    # print(m)
    # x = torch.randn(8, 3, 512, 512)
    # out = m(x)
    # print(out.shape)
