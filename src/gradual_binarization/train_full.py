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


EPOCH = 75
LR_START = 1e-3
LR_END = 1e-3
BATCH_SIZE = 128
WEIGHT_DECAY = 1e-5

BIN_ACTIVE = False


def train(epoch, model, train_loader, optimizer, criterion):
    model.train()
    
    num_data = 0
    num_correct = 0
    sum_loss = 0

    lr_epoch = optimizer.param_groups[0]['lr']
    
    pbar = tqdm(enumerate(train_loader), leave=False, total=len(train_loader))
    for batch_idx, (data, target) in pbar:
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)
        
        # backwarding
        loss = criterion(output, target)
        loss.backward()
        
        optimizer.step()
        
        _, predicted = torch.max(output.data, 1)
        num_data += target.size(0)
        num_correct += (predicted == target).sum().item()
        sum_loss += loss.item() * target.size(0)
    
        avg_loss = sum_loss / num_data
        accuracy = num_correct / num_data

        pbar.set_description(f"EPOCH {epoch:3d}, LR {lr_epoch:.4e} | T LOSS: {avg_loss:.4f}, T ACC: {accuracy*100:.2f}%")
    
    scheduler.step()
    
    avg_loss = sum_loss / num_data
    accuracy = num_correct / num_data

    return avg_loss, accuracy

def validate(epoch, model, test_loader, criterion):
    model.eval()
    
    num_data = 0
    num_correct = 0
    sum_loss = 0

    pbar = tqdm(enumerate(test_loader), leave=False, total=len(test_loader))
    for batch_idx, (data, target) in pbar:
        data, target = Variable(data.cuda()), Variable(target.cuda())
                                    
        output = model(data)
        loss = criterion(output, target)
        
        _, predicted = torch.max(output.data, 1)
        num_data += target.size(0)
        num_correct += (predicted == target).sum().item()
        sum_loss += loss.item() * target.size(0)
    
        avg_loss = sum_loss / num_data
        accuracy = num_correct / num_data

        pbar.set_description(f"EPOCH {epoch:3d} | V LOSS: {avg_loss:.4f}, V ACC: {accuracy*100:.4f}%")

    avg_loss = sum_loss / num_data
    accuracy = num_correct / num_data
    
    return avg_loss, accuracy


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


if __name__=='__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true',
            help='set if only CPU is available')
    parser.add_argument('--data', action='store', default='../data/',
            help='dataset path')
    parser.add_argument('--arch', action='store', default='resnet9',
            help='the architecture for the network: resnet9')
    parser.add_argument('--lr', action='store', default='0.001',
            help='the intial learning rate')
    parser.add_argument('--pretrained', action='store', default=None,
            help='the path to the pretrained model')
    parser.add_argument('--evaluate', action='store_true',
            help='evaluate the model')
    parser.add_argument('--save', action='store_true', default=False,
            help='save the model')
    args = parser.parse_args()
    print('==> Options:',args)

    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    trainloader, testloader = get_CIFAR10_dataset(root=args.data, batch_size=BATCH_SIZE)

    # define classes
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # define the model
    print('==> building model',args.arch,'...')
    if args.arch == 'resnet9':
        model = ResNet9(bin_active=BIN_ACTIVE)
    else:
        raise Exception(args.arch+' is currently not supported')

    # initialize the model
    if not args.pretrained:
        print('==> Initializing model parameters ...')
        best_acc = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                if m.bias is not None:
                    m.bias.data.zero_()
    else:
        print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['best_acc']
        model.load_state_dict(pretrained_model['state_dict'])

    if not args.cpu:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print(model)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,d}")

    # define solver and criterion
    base_lr = float(args.lr)
    param_dict = dict(model.named_parameters())
    params = []

    optimizer = optim.Adam(model.parameters(), lr=LR_START, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss().cuda()
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=(LR_END/LR_START)**(1/EPOCH))

    # define the binarization operator
    bin_op = BinOp(model)

    # start training
    best_acc = 0.0
    for epoch in range(1, EPOCH+1):
        lr_epoch = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, bin_op)
        val_loss, val_acc = validate(epoch, model, testloader, criterion, bin_op)

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        if args.save:
            if is_best:
                torch.save(model.state_dict(), f"saves/{args.arch}/full_best.pth")

        print(f"EPOCH {epoch:3d}/{EPOCH:3d}, LR {lr_epoch:.4e} | T LOSS: {train_loss:.4f}, T ACC: {train_acc*100:.2f}%, V LOSS: {val_loss:.4f}, V ACC: {val_acc*100:.2f}%")
    
    print(f"Best accuracy: {best_acc*100:.2f}%")
    # rename the best model
    if args.save:
        os.rename(f"saves/{args.arch}/full_best.pth", f"saves/{args.arch}/full_best_vacc{int(best_acc*1e+4)}.pth")
