from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import torch
import argparse
import dataloader
from binarizer import BinOp
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from tqdm import tqdm

from models import ResNet9
from torch.autograd import Variable


MODEL_PATHS = [
    './saves/resnet9/xnor_best_vacc8526.pth',
    './saves/resnet9/xnor_best_vacc8550.pth',
]


def get_CIFAR10_dataset(root='../data', batch_size=128, augmentation=True):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=root, train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=root, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    return train_loader, val_loader


models = [ResNet9() for _ in range(len(MODEL_PATHS))]
for i, model_path in enumerate(MODEL_PATHS):
    models[i].cuda()
    models[i] = torch.nn.DataParallel(models[i], device_ids=range(torch.cuda.device_count()))
    models[i].load_state_dict(torch.load(model_path))

bin_ops = [BinOp(model) for model in models]
for bin_op in bin_ops:
    bin_op.binarization()

# Load CIFAR-10 dataset
trainloader, testloader = get_CIFAR10_dataset(root='../data', batch_size=128)

criterion = nn.CrossEntropyLoss().cuda()

num_data = 0
num_correct = 0

pbar = tqdm(enumerate(testloader), leave=False, total=len(testloader))
for batch_idx, (data, target) in pbar:
    data, target = Variable(data.cuda()), Variable(target.cuda())

    prediction = torch.zeros(data.size(0), 10).cuda()

    for model in models:
        model.eval()

        output = model(data)
        probability = nn.functional.softmax(output, dim=1)
        prediction += probability

    _, predicted = torch.max(prediction.data, 1)
    num_data += target.size(0)
    num_correct += (predicted == target).sum().item()

    accuracy = num_correct / num_data

    pbar.set_description(f"V ACC: {accuracy*100:.2f}%")

accuracy = num_correct / num_data

print(f"Ensemble Accuracy: {accuracy*100:.2f}%")


