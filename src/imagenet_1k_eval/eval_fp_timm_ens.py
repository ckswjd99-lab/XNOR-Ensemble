import timm
import torch
import torch.nn as nn
import torchinfo

from tqdm import tqdm

from dataloader import get_imnet1k_dataloader

BATCH_SIZE = 128


# IMNET_DIR = '/data/ImageNet/imagenet1k'
IMNET_DIR = '/data/imagenet'


train_loader, val_loader = get_imnet1k_dataloader(root=IMNET_DIR, batch_size=BATCH_SIZE, augmentation=False)

print("Loaded ImageNet-1k dataset")

model_list = [
    # 'resnet18d.ra2_in1k',
    # 'edgenext_x_small.in1k',
    'edgenext_base.in21k_ft_in1k',
    # 'edgenext_xx_small.in1k',
    # 'mobilevitv2_050.cvnets_in1k',
    # 'mobilevit_xxs.cvnets_in1k',
    # 'dla60x_c.in1k',
]

# criterion = nn.CrossEntropyLoss().cuda()
criterion = nn.NLLLoss().cuda()

# validation
@torch.no_grad()
def validate(test_loader, models, criterion):
    for model in models:
        model.eval()
    
    num_data = 0
    num_correct = 0
    sum_loss = 0
    
    pbar = tqdm(test_loader, leave=False, total=len(test_loader))
    for data, target in pbar:
        data, target = data.cuda(), target.cuda()
        outputs = [nn.Softmax(dim=1)(model(data)) for model in models]
        output = sum(outputs) / len(models)
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


models = [timm.create_model(mname, pretrained=True).cuda() for mname in model_list]

for mname, model in zip(model_list, models):
    model_stat = torchinfo.summary(model, (1, 3, 224, 224), verbose=0)
    num_params = model_stat.total_params
    num_flops = model_stat.total_mult_adds

    print(f"Model: {mname}, Parameters: {num_params:,d}, FLOPs: {num_flops:,d}")

val_loss, val_acc = validate(val_loader, models, criterion)

print(f"ENS | Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")

    
