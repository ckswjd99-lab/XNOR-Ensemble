import timm
import torch
import torch.nn as nn

import torch.quantization as tq

from tqdm import tqdm

from dataloader import get_imnet1k_dataloader


# MODEL_NAME = 'resnet18d.ra2_in1k'
# MODEL_NAME = 'resnet34d.ra2_in1k'
MODEL_NAME = 'resnet152d.ra2_in1k'
BATCH_SIZE = 256

IMNET_DIR = '/data/imagenet'


train_loader, val_loader = get_imnet1k_dataloader(root=IMNET_DIR, batch_size=BATCH_SIZE, augmentation=False)
criterion = nn.CrossEntropyLoss()

# validation
@torch.no_grad()
def validate(test_loader, model, criterion):
    model.eval()
    
    num_data = 0
    num_correct = 0
    sum_loss = 0
    
    pbar = tqdm(test_loader, leave=False, total=len(test_loader))
    for data, target in pbar:
        data, target = data, target
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



model_original = timm.create_model(MODEL_NAME, pretrained=True)
model_original.eval()

model_fp16_1 = timm.create_model(MODEL_NAME, pretrained=True)
model_fp16_1 = tq.quantize_dynamic(model_fp16_1, qconfig_spec={torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
model_fp16_1.eval()

model_fp16_2 = timm.create_model(MODEL_NAME, pretrained=True)
model_fp16_2.eval()

# weights of model_fp16_2 are symmetric to model_fp16_1, centered around model_original
for po, p1, p2 in zip(model_original.parameters(), model_fp16_1.parameters(), model_fp16_2.parameters()):
    p2.data.copy_(2 * po.data - p1.data)

model_fp16_2 = tq.quantize_dynamic(model_fp16_2, qconfig_spec={torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
model_fp16_2.eval()

# validation
sum_loss_orig = 0
sum_loss_single = 0
sum_loss_reg = 0

sum_diff_single = 0
sum_diff_reg = 0

sum_kl_single = 0
sum_kl_reg = 0

num_correct_orig = 0
num_correct_single = 0
num_correct_reg = 0

num_data = 0

with torch.no_grad():
    pbar = tqdm(val_loader, leave=False, total=len(val_loader))
    for data, target in pbar:
        data, target = data, target
        data_half, target_half = data.half(), target.half()

        output_original = model_original(data)
        output_fp16_1 = model_fp16_1(data)
        output_fp16_2 = model_fp16_2(data)
        output_fp16 = (output_fp16_1 + output_fp16_2) / 2

        prob_original = nn.Softmax(dim=1)(output_original)
        prob_fp16_1 = nn.Softmax(dim=1)(output_fp16_1)
        prob_fp16 = nn.Softmax(dim=1)(output_fp16)

        # KL divergence between original and fp16_1
        kl_div = nn.KLDivLoss(reduction='batchmean')(torch.log(prob_original), prob_fp16_1)
        sum_kl_single += kl_div

        # KL divergence between original and fp16
        kl_div_reg = nn.KLDivLoss(reduction='batchmean')(torch.log(prob_original), prob_fp16)
        sum_kl_reg += kl_div_reg

        # output diff between original and fp16
        diff_single = (output_original - output_fp16_1).norm(dim=1).mean().item()
        sum_diff_single += diff_single

        diff_reg = (output_original - output_fp16).norm().item()
        sum_diff_reg += diff_reg

        # loss of original and fp16
        loss_original = criterion(output_original, target)
        loss_fp16_1 = criterion(output_fp16_1, target)
        loss_fp16 = criterion(output_fp16, target)

        sum_loss_orig += loss_original.item()
        sum_loss_single += loss_fp16_1.item()
        sum_loss_reg += loss_fp16.item()

        # accuracy of original and fp16
        _, predicted_original = torch.max(output_original.data, 1)
        _, predicted_fp16_1 = torch.max(output_fp16_1.data, 1)
        _, predicted_fp16 = torch.max(output_fp16.data, 1)

        num_correct_orig += (predicted_original == target).sum().item()
        num_correct_single += (predicted_fp16_1 == target).sum().item()
        num_correct_reg += (predicted_fp16 == target).sum().item()

        num_data += target.size(0)

        pbar.set_description(f'Normalized Diff: {sum_diff_reg/num_data:.4f}, Single Diff: {sum_diff_single/num_data:.4f}, KL: {sum_kl_reg/num_data:.4e}, KL Single: {sum_kl_single/num_data:.4e}')

print(f'Normalized Diff: {sum_diff_reg/num_data:.4f}, Single Diff: {sum_diff_single/num_data:.4f}, KL: {sum_kl_reg/num_data:.4e}, KL Single: {sum_kl_single/num_data:.4e}')
print(f'Loss Original: {sum_loss_orig/num_data:.4e}, Loss Single: {sum_loss_single/num_data:.4e}, Loss Reg: {sum_loss_reg/num_data:.4e}')
print(f'Accuracy Original: {num_correct_orig/num_data:.4f}, Accuracy Single: {num_correct_single/num_data:.4f}, Accuracy Reg: {num_correct_reg/num_data:.4f}')