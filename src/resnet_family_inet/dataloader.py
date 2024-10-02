import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

IMNET_DIR = '/data/ImageNet/imagenet1k'
IMNET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMNET_DEFAULT_STD = (0.229, 0.224, 0.225)


def get_imnet1k_loader(root=IMNET_DIR, batch_size=128, augmentation=False):
    train_path = root + '/train'
    test_path = root + '/val'

    aug_basic = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMNET_DEFAULT_MEAN, IMNET_DEFAULT_STD)
    ])
    aug_simple = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMNET_DEFAULT_MEAN, IMNET_DEFAULT_STD)
    ])

    train_dataset = datasets.ImageFolder(train_path, aug_simple if augmentation else aug_basic)
    test_dataset = datasets.ImageFolder(test_path, aug_basic)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    return train_loader, test_loader

