# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import math
import os
import json

import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

# IMNET_DIR = '/data/ImageNet/imagenet1k'
IMNET_DIR = '/data/imagenet'
NUM_WORKERS = 16


def get_CIFAR10_dataset(root='./data', batch_size=128, hard_aug=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if hard_aug:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=root, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=batch_size, shuffle=False,
            num_workers=1, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=root, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=batch_size, shuffle=False,
            num_workers=1, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=root, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True)

    return train_loader, val_loader


def get_imnet1k_dataloader(root=IMNET_DIR, batch_size=128, augmentation="noaug"):
    dataset_train, nb_classes = build_dataset(is_train=True, input_size=224, augmentation=augmentation)
    dataset_val, _ = build_dataset(is_train=False, input_size=224, augmentation=augmentation)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size,
        shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True, drop_last=False
    )

    return train_loader, val_loader


def build_dataset(is_train, input_size=224, augmentation="noaug"):
    transform = build_transform(is_train, input_size, augmentation)

    root = os.path.join(IMNET_DIR, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)
    nb_classes = 1000

    return dataset, nb_classes


def build_transform(is_train, input_size=224, augmentation="noaug"):
    resize_im = input_size > 32
    if is_train:
        if augmentation == "noaug":
            transform = transforms.Compose([
                transforms.RandomResizedCrop(input_size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
            return transform
        else:
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=input_size,
                is_training=True,
                color_jitter=0.4,
                auto_augment=augmentation,
                interpolation='bicubic',
                re_prob=0.25,
                re_mode='pixel',
                re_count=1,
            )
            if not resize_im:
                # replace RandomResizedCropAndInterpolation with
                # RandomCrop
                transform.transforms[0] = transforms.RandomCrop(
                    input_size, padding=4)
            return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
