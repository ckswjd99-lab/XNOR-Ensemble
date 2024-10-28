import torch
import torch.nn as nn

import timm
import time

from tqdm import tqdm

from binarizer import BinOp
from dataloader import get_imnet1k_dataloader


CHECKPOINT_PATHS = [
    './saves/fs_resnet18d.ra2_in1k_bw_best.pth',
    './saves/fs_resnet18d.ra2_in1k_bw_e4_best.pth',
    # './saves/ft_resnet18d_ra2_in1k_wb_best.pth',
    # './saves/ft_resnet18d.ra2_in1k_bw_e6_best.pth',
    # './saves/ft_resnet18d.ra2_in1k_bw_e4toe6_best.pth'
]
BATCH_SIZE = 128

@torch.no_grad()
def validate(epoch, model_list, test_loader, criterion):
    for model in model_list:
        model.eval()
    
    num_data = 0
    num_correct = 0
    sum_loss = 0

    pbar = tqdm(enumerate(test_loader), leave=False, total=len(test_loader))
    for batch_idx, (data, target) in pbar:

        data, target = data.cuda(), target.cuda()
                                    
        outputs = [nn.Softmax(dim=1)(model(data)) for model in model_list]
        output = sum(outputs) / len(model_list)
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


models = [timm.create_model('resnet18d.ra2_in1k', pretrained=False).cuda() for _ in range(len(CHECKPOINT_PATHS))]

num_params = sum(p.numel() for p in models[0].parameters())
print(f"Number of parameters: {num_params:,d}")

bin_ops = [BinOp(model) for model in models]

# criterion = nn.CrossEntropyLoss().cuda()
criterion = nn.NLLLoss().cuda()

for model, checkpoint, bin_op in zip(models, CHECKPOINT_PATHS, bin_ops):
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model'])

    bin_op.binarization()

train_loader, val_loader = get_imnet1k_dataloader(batch_size=BATCH_SIZE, augmentation=False)

val_loss, val_acc = validate(0, models, val_loader, criterion)

print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")