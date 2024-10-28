import torch
import torch.nn as nn
import torchinfo

import timm
import time

from binarizer import BinOp
from dataloader import get_imnet1k_dataloader
from trainer import train, validate


# RESUME_CHECKPOINT = './saves/ft_resnet18d_ra2_in1k_xnorized_best.pth'
MODEL_NAME = 'resnet18d.ra2_in1k'
BATCH_SIZE = 128


model = timm.create_model(MODEL_NAME, pretrained=False).cuda()
print(model)

model_stat = torchinfo.summary(model, (1, 3, 224, 224), verbose=0)
print(f"Model: {model.__class__.__name__} | Parameters: {model_stat.total_params:,d} | FLOPs: {model_stat.total_mult_adds:,d}")

num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params:,d}")

bin_op = BinOp(model)
print(f'Number of binary convolutions: {bin_op.countBinaryParams():,d}')

exit(0)

criterion = nn.CrossEntropyLoss().cuda()

checkpoint = torch.load(RESUME_CHECKPOINT)
model.load_state_dict(checkpoint['model'])
best_vacc = checkpoint['best_vacc']
start_epoch = checkpoint['epoch']
print(f"Using checkpoint from epoch {start_epoch}, with best validation accuracy: {best_vacc:.4f}")


train_loader, val_loader = get_imnet1k_dataloader(batch_size=BATCH_SIZE, augmentation=False)

val_loss, val_acc = validate(0, model, val_loader, criterion, bin_op)

print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")