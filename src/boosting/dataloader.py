import os
import torch
import pickle
import numpy
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class dataset():
    def __init__(self, root=None, train=True):
        self.root = root
        self.train = train
        self.transform = transforms.ToTensor()
        if self.train:
            train_data_path = os.path.join(root, 'train_data')
            train_labels_path = os.path.join(root, 'train_labels')
            self.train_data = numpy.load(open(train_data_path, 'r'))
            self.train_data = torch.from_numpy(self.train_data.astype('float32'))
            self.train_labels = numpy.load(open(train_labels_path, 'r')).astype('int')
        else:
            test_data_path = os.path.join(root, 'test_data')
            test_labels_path = os.path.join(root, 'test_labels')
            self.test_data = numpy.load(open(test_data_path, 'r'))
            self.test_data = torch.from_numpy(self.test_data.astype('float32'))
            self.test_labels = numpy.load(open(test_labels_path, 'r')).astype('int')

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]


        return img, target

def get_CIFAR10_dataset(root='./data', batch_size=128, augmentation=True):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=root, train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=root, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    return train_loader, val_loader