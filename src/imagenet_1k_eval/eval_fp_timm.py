import timm
import torch
import torch.nn as nn

from tqdm import tqdm

from dataloader import get_imnet1k_dataloader

BATCH_SIZE = 128

'''
+======================================================================================+
|                                IMAGENET-1K VALIDATION                                |
|--------------------------------------------------------------------------------------|
|  Model (#PARAM)                                               |  V LOSS  |   V ACC   |
+======================================================================================+
|  tiny_vit_5m_224.dist_in22k_ft_in1k             (  5,392,764) |  0.8266  |  80.72 %  |
|  tiny_vit_11m_224.dist_in22k_ft_in1k            ( 10,996,972) |  0.7271  |  83.19 %  |
|  tiny_vit_11m_224.in1k                          ( 10,996,972) |  0.9834  |  81.45 %  |
|  tiny_vit_21m_224.dist_in22k_ft_in1k            ( 21,198,568) |  0.6422  |  84.85 %  |
|  tiny_vit_21m_224.in1k                          ( 21,198,568) |  0.8458  |  83.09 %  |
|  deit_small_distilled_patch16_224.fb_in1k       ( 22,436,432) |  0.7511  |  81.18 %  |
|  deit_base_patch16_224.fb_in1k                  ( 86,567,656) |  0.8198  |  81.80 %  |
|  deit_base_distilled_patch16_224.fb_in1k        ( 87,338,192) |  0.6839  |  83.33 %  |
|  deit3_small_patch16_224.fb_in22k_ft_in1k       ( 22,059,496) |  0.7308  |  82.76 %  |
|  deit3_small_patch16_224.fb_in1k                ( 22,059,496) |  0.8930  |  81.31 %  |
|  deit3_medium_patch16_224.fb_in22k_ft_in1k      ( 38,849,512) |  0.6892  |  84.16 %  |
|  deit3_medium_patch16_224.fb_in1k               ( 38,849,512) |  0.7686  |  82.90 %  |
|  deit3_base_patch16_224.fb_in22k_ft_in1k        ( 86,585,320) |  0.6194  |  85.49 %  |
|  deit3_base_patch16_224.fb_in1k                 ( 86,585,320) |  0.7365  |  83.68 %  |
|  swin_tiny_patch4_window7_224.ms_in22k_ft_in1k  ( 28,288,354) |  0.8108  |  80.91 %  |
|  swin_tiny_patch4_window7_224.ms_in1k           ( 28,288,354) |  0.8143  |  81.18 %  |
|  swin_small_patch4_window7_224.ms_in22k_ft_in1k ( 49,606,258) |  0.6829  |  83.26 %  |
|  swin_small_patch4_window7_224.ms_in1k          ( 49,606,258) |  0.7343  |  83.21 %  |
|  swin_base_patch4_window7_224.ms_in22k_ft_in1k  ( 87,768,224) |  0.6455  |  85.16 %  |
|  swin_base_patch4_window7_224.ms_in1k           ( 87,768,224) |  0.7402  |  83.42 %  |
|  swinv2_cr_tiny_ns_224.sw_in1k                  ( 28,333,468) |  0.8331  |  81.60 %  |
|  swinv2_cr_small_ns_224.sw_in1k                 ( 49,696,444) |  0.7531  |  83.41 %  |
|  swinv2_cr_small_224.sw_in1k                    ( 49,695,100) |  0.7927  |  82.90 %  |
|  swin_s3_tiny_224.ms_in1k                       ( 28,328,674) |  0.7743  |  82.03 %  |
|  swin_s3_small_224.ms_in1k                      ( 49,737,298) |  0.7240  |  83.66 %  |
|  swin_s3_base_224.ms_in1k                       ( 71,125,762) |  0.7035  |  83.94 %  |
+======================================================================================+

'''


# IMNET_DIR = '/data/ImageNet/imagenet1k'
IMNET_DIR = '/data/imagenet'


train_loader, val_loader = get_imnet1k_dataloader(root=IMNET_DIR, batch_size=BATCH_SIZE, augmentation=False)

print("Loaded ImageNet-1k dataset")

model_list = [
    # 'tiny_vit_5m_224.dist_in22k_ft_in1k',
    # 'tiny_vit_11m_224.dist_in22k_ft_in1k',
    # 'tiny_vit_11m_224.in1k',
    # 'tiny_vit_21m_224.dist_in22k_ft_in1k',
    # 'tiny_vit_21m_224.in1k',
    # 'deit_small_distilled_patch16_224.fb_in1k',
    # 'deit_base_patch16_224.fb_in1k',
    # 'deit_base_distilled_patch16_224.fb_in1k',
    # 'deit3_small_patch16_224.fb_in22k_ft_in1k',
    # 'deit3_small_patch16_224.fb_in1k',
    # 'deit3_medium_patch16_224.fb_in22k_ft_in1k',
    # 'deit3_medium_patch16_224.fb_in1k',
    # 'deit3_base_patch16_224.fb_in22k_ft_in1k',
    # 'deit3_base_patch16_224.fb_in1k',
    'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k',
    'swin_tiny_patch4_window7_224.ms_in1k',
    'swin_small_patch4_window7_224.ms_in22k_ft_in1k',
    'swin_small_patch4_window7_224.ms_in1k',
    'swin_base_patch4_window7_224.ms_in22k_ft_in1k',
    'swin_base_patch4_window7_224.ms_in1k',
    'swinv2_cr_tiny_ns_224.sw_in1k',
    'swinv2_cr_small_ns_224.sw_in1k',
    'swinv2_cr_small_224.sw_in1k',
    'swin_s3_tiny_224.ms_in1k',
    'swin_s3_small_224.ms_in1k',
    'swin_s3_base_224.ms_in1k',

]

model_vloss = []
model_vacc = []
model_nparams = []

criterion = nn.CrossEntropyLoss().cuda()

# validation
@torch.no_grad()
def validate(test_loader, model, criterion):
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


print("+====================================================================================+")
print("|                               IMAGENET-1K VALIDATION                               |")
print("|------------------------------------------------------------------------------------|")
print("|  Model (#PARAM)                                             |  V LOSS  |   V ACC   |")
print("+====================================================================================+")

for mname in model_list:
    model = timm.create_model(mname, pretrained=True, num_classes=1000).cuda()
    nparams = sum(p.numel() for p in model.parameters())

    val_loss, val_acc = validate(val_loader, model, criterion)

    model_vloss.append(val_loss)
    model_vacc.append(val_acc)

    num_params = sum(p.numel() for p in model.parameters())
    model_nparams.append(num_params)

    print(f'|  {mname:44s} ({nparams:11,d}) |  {val_loss:.4f}  |  {val_acc*100:.2f} %  |')

print("+====================================================================================+")