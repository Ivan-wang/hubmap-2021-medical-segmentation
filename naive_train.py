import segmentation_models_pytorch as smp
import torch.nn as nn
import torch
from catalyst.utils.metrics import dice

from tqdm import tqdm
from torch import optim
from dataset import load_train_data

train, val = load_train_data()

model = smp.Unet(encoder_name='se_resnet50', in_channels=3, classes=1)
model = model.cuda()

learning_rate = 1e-3
decay = 3e-5
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=64)

loss_fn = nn.BCEWithLogitsLoss()
metrics = lambda out, t: dice(out, t, threshold=0.5)

num_epoch = 8
batch_size = 17

avg_loss = None
avg_metric = None

for epoch in range(num_epoch):
    loss_tot = 0
    metrics_tot = 0
    model.train()
    for batch in tqdm(train):
        inputs = batch['image']
        targets = batch['mask']

        inputs = inputs.cuda()
        inputs.requires_grad_(True)
        targets = targets.cuda()

        optimizer.zero_grad()
        out = model(inputs)
        loss = loss_fn(out, targets)

        loss.backward()
        optimizer.step()

        m = metrics(out.detach(), targets)

        loss_tot += loss.detach() * batch_size
        metrics_tot += m * batch_size
    print(f'Training Epoch {epoch+1}, Avg Loss {loss_tot / (len(train)*batch_size):.6f}, \
        Avg Dice {metrics_tot / (len(train) * batch_size):.6f}')

    scheduler.step()

    loss_tot = 0
    metrics_tot = 0
    model.eval()
    with torch.no_grad():
        for batch in val:
            inputs = batch['image']
            targets = batch['mask']

            inputs = inputs.cuda()
            inputs.requires_grad_(False)
            targets = targets.cuda()

            out = model(inputs)
            loss = loss_fn(out, targets)
            m = metrics(out, targets)

            loss_tot += loss * batch_size
            metrics_tot += m * batch_size
        print(f'Validation Epoch {epoch+1}, Avg Loss {loss_tot / (len(val)*batch_size):.6f}, \
            Avg Dice {metrics_tot / (len(val) * batch_size):.6f}')
        val_m = metrics_tot / (len(val) * batch_size)
        model_name = 'model_{}_{:.4f}.pth'.format(epoch+1, val_m)
        torch.save({'state_dict': model.state_dict()}, model_name)
        # if avg_metric is None or avg_metric < metrics_tot / (batch_size * len(val)):
            # avg_metric = metrics_tot / (len(val) * batch_size)
            # print('Saving Best Model...')
            # torch.save({'state_dict': model.state_dict()}, 'best_model_bceloss.pth'

