import torch
import torch.nn as nn
import torch.nn.functional as F

import time
from tqdm import tqdm
import os

from binarizer import BinOp
from dataloader import get_CIFAR10_dataset
from resnet import resnet20, resnet110
from resnet_custom import ResNet9
from trainer import validate_single, validate_multi_feature, validate_multi_hardvote, validate_multi_softvote

BATCH_SIZE = 128
# PREFIX = 'resnet20_bw_dist_naive_'
PREFIX = 'resnet9_bw_dist_naive_'
COLLEAGUES_CKPTS = ['saves/' + f for f in os.listdir('saves') if f.startswith(PREFIX) and f.endswith("_final.pth")]
VALIDATE_EACH = True
# MODEL_ARCH = resnet20
MODEL_ARCH = ResNet9

print(f"Colleagues: {COLLEAGUES_CKPTS}")
# MODEL_ARCH = resnet110


train_loader, val_loader = get_CIFAR10_dataset(root='../data', batch_size=BATCH_SIZE)

criterion_xent = nn.CrossEntropyLoss().cuda()
# criterion_kld = nn.KLDivLoss(reduction='batchmean').cuda()

# LOAD COLLEAGUES
colleagues = [MODEL_ARCH().cuda() for _ in COLLEAGUES_CKPTS]
colleagues_binops = [BinOp(colleague) for colleague in colleagues]

for colleague, ckpt, binop in zip(colleagues, COLLEAGUES_CKPTS, colleagues_binops):
    colleague_ckpt = torch.load(ckpt, weights_only=True)
    colleague.load_state_dict(colleague_ckpt)
    binop.binarization()

    if VALIDATE_EACH:
        colleague_vloss, colleague_vacc = validate_single(0, colleague, val_loader, criterion_xent, None)

        print(f"Colleague {ckpt}: V LOSS: {colleague_vloss:.4f}, V ACC: {colleague_vacc*100:.2f}%")

    # binop.restore()

ensembled_vloss, ensembled_vacc = validate_multi_feature(0, colleagues, val_loader, criterion_xent, None)
_, ensembled_vacc_hardvote = validate_multi_hardvote(0, colleagues, val_loader, criterion_xent, None)
_, ensembled_vacc_softvote = validate_multi_softvote(0, colleagues, val_loader, criterion_xent, None)
print(f"Ensembled: V LOSS: {ensembled_vloss:.4f}, V ACC (feature): {ensembled_vacc*100:.2f}%, V ACC (hardvote): {ensembled_vacc_hardvote*100:.2f}%, V ACC (softvote): {ensembled_vacc_softvote*100:.2f}%")

