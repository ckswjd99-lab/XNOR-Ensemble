import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

from tqdm import tqdm

from dataloader import get_imnet1k_loader
from models import ResNet18

EPOCH = 200
BATCH_SIZE = 128
LR_START = 1e-3
LR_END = 1e-6

train_loader, val_loader = get_imnet1k_loader(batch_size=BATCH_SIZE, augmentation=False)

print("Loaded ImageNet-1k dataset")

model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1').cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR_START)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCH, eta_min=LR_END)

# validation
@torch.no_grad()
def validate(model, test_loader):
    model.eval()
    
    num_data = 0
    num_correct = 0
    sum_loss = 0
    
    pbar = tqdm(test_loader, leave=False, total=len(test_loader))
    for data, target in pbar:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        
        _, predicted = torch.max(output.data, 1)
        num_data += target.size(0)
        num_correct += (predicted == target).sum().item()
        sum_loss += loss.item() * target.size(0)

        avg_loss = sum_loss / num_data
        accuracy = num_correct / num_data

        pbar.set_description(f'Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    avg_loss = sum_loss / num_data
    accuracy = num_correct / num_data
    return avg_loss, accuracy

# validate
val_loss, val_acc = validate(model, val_loader)

print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')