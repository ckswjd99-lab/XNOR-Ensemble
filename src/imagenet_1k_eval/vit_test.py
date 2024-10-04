import torch
import torch.nn as nn

from tqdm import tqdm

from transformers import ViTForImageClassification, DeiTForImageClassificationWithTeacher
from transformers import MobileViTImageProcessor, MobileViTForImageClassification

from dataloader import get_imnet1k_dataloader

EPOCH = 200
BATCH_SIZE = 128
LR_START = 1e-3
LR_END = 1e-6

IMNET_DIR = '/data/ImageNet/imagenet1k'

feature_extractor = MobileViTImageProcessor.from_pretrained("apple/deeplabv3-mobilevit-small")
model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small").cuda()

train_loader, val_loader = get_imnet1k_dataloader(root=IMNET_DIR, batch_size=BATCH_SIZE, augmentation=False)

input_temp, target_temp = next(iter(val_loader))
input_temp, target_temp = input_temp.cuda(), target_temp.cuda()

criterion = nn.CrossEntropyLoss().cuda()

# inference
output = model(input_temp)


print(output.logits.shape)


# validate
model.eval()

num_data = 0
num_correct = 0
sum_loss = 0

with torch.no_grad():
    pbar = tqdm(enumerate(val_loader), total=len(val_loader), position=0, leave=True)
    for i, (input, target) in pbar:
        input, target = input.cuda(), target.cuda()

        # input = feature_extractor(images=input, return_tensors="pt")

        output = model(input)
        loss = criterion(output.logits, target)

        sum_loss += loss.item()
        num_data += input.size(0)
        num_correct += (output.logits.argmax(dim=1) == target).sum().item()

        pbar.set_description(f"Validation Loss: {sum_loss / num_data:.4f}, Validation Accuracy: {num_correct / num_data:.4f}")


print(f"Validation Loss: {sum_loss / num_data:.4f}")
print(f"Validation Accuracy: {num_correct / num_data:.4f}")
    