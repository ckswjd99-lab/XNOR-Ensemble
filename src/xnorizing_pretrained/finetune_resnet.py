import torch
import torch.nn as nn

import timm

from xnorizer import xnorize_conv2d
from binarizer import BinOp
from dataloader import get_imnet1k_dataloader
from trainer import train, validate

BATCH_SIZE = 128
EPOCHS = 200
LR_START = 1e-2
LR_END = 1e-5

model = timm.create_model('resnet18d.ra2_in1k', pretrained=True).cuda()
model = xnorize_conv2d(model).cuda()

print(model)

num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params:,d}")

bin_op = BinOp(model)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LR_START, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR_END)

train_loader, val_loader = get_imnet1k_dataloader(batch_size=BATCH_SIZE, augmentation=False)

for epoch in range(EPOCHS):
    train_loss, train_acc = train(epoch, model, train_loader, optimizer, criterion, bin_op)
    val_loss, val_acc = validate(epoch, model, val_loader, criterion, bin_op)
    scheduler.step()

    print(f"EPOCH {epoch:3d}, LR: {scheduler.get_last_lr()[0]:.4e} | T LOSS: {train_loss:.4f}, T ACC: {train_acc*100:.2f}%, V LOSS: {val_loss:.4f}, V ACC: {val_acc:.4f}% |")