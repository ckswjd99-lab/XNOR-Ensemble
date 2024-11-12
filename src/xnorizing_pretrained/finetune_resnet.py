import torch
import torch.nn as nn

import timm
import time

from xnorizer import xnorize_conv2d
from binarizer import BinOp
from dataloader import get_imnet1k_dataloader
from trainer import train, validate

BATCH_SIZE = 128
EPOCHS = 200
LR_START = 1e-3
LR_END = 1e-5
# MODEL_NAME = 'resnet18d.ra2_in1k'
MODEL_NAME = 'resnet50.a1_in1k'
# CKPT_NAME = f'ft_{MODEL_NAME}_bw.pth'
CKPT_NAME = f'ft_{MODEL_NAME}_bw_03.pth'

RESUME_CHECKPOINT = None
# RESUME_CHECKPOINT = './saves/ft_resnet18d_ra2_in1k_xnorized.pth'

model = timm.create_model(MODEL_NAME, pretrained=True).cuda()
# model = xnorize_conv2d(model).cuda()

print(model)

num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params:,d}")

bin_op = BinOp(model)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LR_START, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(LR_END/LR_START)**(1/(EPOCHS-1)))

train_loader, val_loader = get_imnet1k_dataloader(batch_size=BATCH_SIZE, augmentation=False)

best_vacc = 0
start_epoch = 0

if RESUME_CHECKPOINT:
    checkpoint = torch.load(RESUME_CHECKPOINT)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    best_vacc = checkpoint['best_vacc']
    start_epoch = checkpoint['epoch']

for epoch in range(start_epoch, EPOCHS):
    start_time = time.time()

    train_loss, train_acc = train(epoch, model, train_loader, optimizer, criterion, bin_op)
    val_loss, val_acc = validate(epoch, model, val_loader, criterion, bin_op)
    scheduler.step()

    is_best = val_acc > best_vacc
    best_vacc = max(val_acc, best_vacc)

    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_vacc': best_vacc
    }
    torch.save(checkpoint, f"saves/{CKPT_NAME}")

    if is_best:
        torch.save(checkpoint, f"saves/{CKPT_NAME.replace('.pth', '_best.pth')}")


    end_time = time.time()
    print(f"EPOCH {epoch:3d}, LR: {scheduler.get_last_lr()[0]:.4e} | T LOSS: {train_loss:.4f}, T ACC: {train_acc*100:.2f}%, V LOSS: {val_loss:.4f}, V ACC: {val_acc*100:.2f}% | ETA: {int(end_time-start_time) // 60:8,d} min")