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

from torch.utils.data import Dataset, DataLoader, random_split


from tqdm import tqdm

from models import ResNet9
from torch.autograd import Variable


EPOCH = 100
BATCH_SIZE = 32
LR_START = 1e-3
LR_END = 1e-5

NUM_LEARNERS = 4


def train(epoch, model, train_loader, optimizer, criterion, bin_op):
    model.train()
    
    num_data = 0
    num_correct = 0
    sum_loss = 0

    lr_epoch = optimizer.param_groups[0]['lr']
    
    pbar = tqdm(enumerate(train_loader), leave=False, total=len(train_loader))
    for batch_idx, (data, target) in pbar:
        # process the weights including binarization
        bin_op.binarization()
        
        # forwarding
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)
        
        # backwarding
        loss = criterion(output, target)
        loss.backward()
        
        # restore weights
        bin_op.restore()
        bin_op.updateBinaryGradWeight()
        
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

def validate(epoch, model, test_loader, criterion, bin_op):
    model.eval()
    bin_op.binarization()
    
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

    bin_op.restore()
    avg_loss = sum_loss / num_data
    accuracy = num_correct / num_data
    
    return avg_loss, accuracy


def get_CIFAR10_dataset(root='../data', batch_size=128, num_split=4):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    train_dataset = datasets.CIFAR10(root=root, train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
    
    splitted_train_datasets = random_split(train_dataset, [len(train_dataset)//num_split]*num_split)
    
    test_dataset = datasets.CIFAR10(root=root, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    
    train_loaders = [
        DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        for dataset in splitted_train_datasets
    ]

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loaders, test_loader


if __name__=='__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true',
            help='set if only CPU is available')
    parser.add_argument('--data', action='store', default='./data/',
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

    train_loaders, test_loader = get_CIFAR10_dataset(root=args.data, batch_size=BATCH_SIZE, num_split=NUM_LEARNERS)

    # define the model
    models = [ResNet9() for _ in range(NUM_LEARNERS)]

    # initialize the model
    best_acc = 0
    for model in models:
        model.cuda()
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                if m.bias is not None:
                    m.bias.data.zero_()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,d}")
    print()

    binarizers = [BinOp(model) for model in models]
    optimizers = [optim.Adam(model.parameters(), lr=LR_START, weight_decay=0.00001) for model in models]
    schedulers = [optim.lr_scheduler.ExponentialLR(optimizer, gamma=(LR_END/LR_START)**(1/EPOCH)) for optimizer in optimizers]

    criterion = nn.CrossEntropyLoss()

    # start training
    for model_idx, train_loader, model, bin_op, optimizer, scheduler in zip(range(NUM_LEARNERS), train_loaders, models, binarizers, optimizers, schedulers):
        print(f"Training learner {model_idx:02d}")
        
        best_acc = 0.0

        for epoch in range(1, EPOCH+1):
            lr_epoch = optimizer.param_groups[0]['lr']

            train_loss, train_acc = train(epoch, model, train_loader, optimizer, criterion, bin_op)
            val_loss, val_acc = validate(epoch, model, test_loader, criterion, bin_op)

            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)

            if args.save and is_best:
                torch.save(model.state_dict(), f"saves/{args.arch}/learner{model_idx:02d}_model_best.pth")

            print(
                f"EPOCH {epoch:3d}/{EPOCH:3d}, LR {lr_epoch:.4e} | T LOSS: {train_loss:.4f}, T ACC: {train_acc*100:.2f}%, V LOSS: {val_loss:.4f}, V ACC: {val_acc*100:.2f}% |"
                + (" *" if is_best else "")
            )
        
        print(f"Best accuracy: {best_acc*100:.2f}%")

        if args.save:
            model.load_state_dict(torch.load(f"saves/{args.arch}/learner{model_idx:02d}_model_best.pth"))
            torch.save(model.state_dict(), f"saves/{args.arch}/learner{model_idx:02d}_final_vacc{int(best_acc*1e4)}.pth")
        
        print()

    # evaluate each model
    for model_idx, model, bin_op in zip(range(NUM_LEARNERS), models, binarizers):
        print(f"Evaluating learner {model_idx:02d}")
        
        bin_op.binarization()

        model.eval()
        num_data = 0
        num_correct = 0

        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            
            output = torch.nn.functional.softmax(model(data), dim=1)
            _, predicted = torch.max(output.data, 1)
            num_data += target.size(0)
            num_correct += (predicted == target).sum().item()

        accuracy = num_correct / num_data
        print(f"Learner {model_idx:02d} accuracy: {accuracy*100:.2f}%")

        bin_op.restore()

    # ensemble and evaluate
    print(f"Ensemble {NUM_LEARNERS} learners")

    num_data = 0
    num_correct = 0

    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = Variable(data.cuda()), Variable(target.cuda())
        
        outputs = [torch.nn.functional.softmax(model(data), dim=1) for model in models]
        output = sum(outputs)
        
        _, predicted = torch.max(output.data, 1)
        num_data += target.size(0)
        num_correct += (predicted == target).sum().item()

    accuracy = num_correct / num_data
    print(f"Ensemble accuracy: {accuracy*100:.2f}%")
    