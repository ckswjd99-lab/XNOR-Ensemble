import timm
import torch
import torch.nn as nn
import torchinfo

from tqdm import tqdm

from dataloader import get_imnet1k_dataloader

BATCH_SIZE = 256

IMNET_DIR = '/data/imagenet'


train_loader, val_loader = get_imnet1k_dataloader(root=IMNET_DIR, batch_size=BATCH_SIZE, augmentation=False)
criterion = nn.NLLLoss().cuda()

resnet18_family = {
    'resnet18d.ra2_in1k': 73.794,
    'resnet18.fb_swsl_ig1b_ft_in1k': 73.288,
    # 'resnet18.a1_in1k': 73.164,
    # 'resnet18.fb_ssl_yfcc100m_ft_in1k': 72.636,
    # 'resnet18.a2_in1k': 72.362,
    # 'resnet18.a1_in1k': 71.488,
    # 'resnet18.gluon_in1k': 70.844,
    # 'resnet18.a2_in1k': 70.634,
    # 'resnet18.tv_in1k': 69.752,
    # 'resnet18.a3_in1k': 68.246,
}

asymm_family = {
    'resnetrs152.tf_in1k': 83.71,
    'resnet101d.ra2_in1k': 83.018,
    'resnet51q.ra2_in1k': 82.356,
}


# configuration
@torch.no_grad()
def config_models(model_list):
    total_params = 0
    total_macs = 0
    for model in model_list:
        model.eval()

        model_stat = torchinfo.summary(model, (1, 3, 224, 224), verbose=0)

        total_params += model_stat.total_params
        total_macs += model_stat.total_mult_adds

        print(f"Model: {model.__class__.__name__} | Parameters: {model_stat.total_params:,d} | FLOPs: {model_stat.total_mult_adds:,d}")
    
    print(f'Total parameters: {total_params:,d}')
    print(f'Total FLOPs: {total_macs:,d}')


# validation
@torch.no_grad()
def validate_list(test_loader, model_list, criterion):
    for model in model_list:
        model.eval()
    
    num_data = 0
    num_correct = 0
    sum_loss = 0
    
    pbar = tqdm(test_loader, leave=False, total=len(test_loader))
    for data, target in pbar:
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

        pbar.set_description(f'Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    avg_loss = sum_loss / num_data
    accuracy = num_correct / num_data
    
    return avg_loss, accuracy


models_sym = [timm.create_model(model, pretrained=True).cuda() for model in resnet18_family.keys()]
print(list(resnet18_family.keys()))
print(len(models_sym))

config_models(models_sym)

sym_loss, sym_accuracy = validate_list(val_loader, models_sym, criterion)

print(f'Symmetric ensemble loss: {sym_loss:.4f}, accuracy: {sym_accuracy:.4f}')


models_asym = [timm.create_model(model, pretrained=True).cuda() for model in asymm_family.keys()]

config_models(models_asym)

asym_loss, asym_accuracy = validate_list(val_loader, models_asym, criterion)

print(f'Asymmetric ensemble loss: {asym_loss:.4f}, accuracy: {asym_accuracy:.4f}')