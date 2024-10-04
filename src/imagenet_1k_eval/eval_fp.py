import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import DeiTForImageClassificationWithTeacher, MobileViTForImageClassification

from tqdm import tqdm

from dataloader import get_imnet1k_dataloader

'''
+========================================================+
|                 IMAGENET-1K VALIDATION                 |
|--------------------------------------------------------|
|  Model (#PARAM)                 |  V LOSS  |   V ACC   |
|========================================================|
|  MobNetV2         (  3,504,872) |  1.7921  |  71.64 %  |
|  MobNetV3_S       (  2,542,856) |  1.3532  |  67.31 %  |
|  MobNetV3_L       (  5,483,032) |  1.1802  |  75.14 %  |
|  VGG11_BN         (132,868,840) |  1.2003  |  70.24 %  |
|  VGG13_BN         (133,053,736) |  1.1532  |  71.36 %  |
|  VGG16_BN         (138,365,992) |  1.0750  |  73.06 %  |
|  VGG19_BN         (143,678,248) |  1.0488  |  74.00 %  |
|  ResNet18         ( 11,689,512) |  1.2535  |  69.56 %  |
|  ResNet34         ( 21,797,672) |  1.0856  |  73.21 %  |
|  ResNet50         ( 25,557,032) |  1.4217  |  80.14 %  |
|  ResNet101        ( 44,549,160) |  0.9005  |  81.47 %  |
|  ResNet152        ( 60,192,808) |  0.8723  |  82.20 %  |
|  ViT-B_16         ( 86,567,656) |  0.8415  |  80.90 %  |
|  ViT-B_32         ( 88,224,232) |  1.0795  |  75.73 %  |
|  ViT-L_16         (304,326,632) |  0.9129  |  79.52 %  |
|  ViT-L_32         (306,535,400) |  1.0283  |  76.90 %  |
|  DeiT-tiny(D)     (  5,910,800) |  1.0214  |  74.52 %  |
|  DeiT-small(D)    ( 22,436,432) |  0.7511  |  81.18 %  |
|  DeiT-base(D)     ( 87,338,192) |  0.6839  |  83.34 %  |
+========================================================+

'''



EPOCH = 200
BATCH_SIZE = 128
LR_START = 1e-3
LR_END = 1e-6

IMNET_DIR = '/data/ImageNet/imagenet1k'


train_loader, val_loader = get_imnet1k_dataloader(root=IMNET_DIR, batch_size=BATCH_SIZE, augmentation=False)

print("Loaded ImageNet-1k dataset")


model_list = [
    models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).cuda(),
    models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).cuda(),
    models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT).cuda(),
    models.vgg11_bn(weights=models.VGG11_BN_Weights.DEFAULT).cuda(),
    models.vgg13_bn(weights=models.VGG13_BN_Weights.DEFAULT).cuda(),
    models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT).cuda(),
    models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT).cuda(),
    models.resnet18(weights=models.ResNet18_Weights.DEFAULT).cuda(),
    models.resnet34(weights=models.ResNet34_Weights.DEFAULT).cuda(),
    models.resnet50(weights=models.ResNet50_Weights.DEFAULT).cuda(),
    models.resnet101(weights=models.ResNet101_Weights.DEFAULT).cuda(),
    models.resnet152(weights=models.ResNet152_Weights.DEFAULT).cuda(),
    models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT).cuda(),
    models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT).cuda(),
    models.vit_l_16(weights=models.ViT_L_16_Weights.DEFAULT).cuda(),
    models.vit_l_32(weights=models.ViT_L_32_Weights.DEFAULT).cuda(),
]

model_deit_list = [
    DeiTForImageClassificationWithTeacher.from_pretrained("facebook/deit-tiny-distilled-patch16-224").cuda(),
    DeiTForImageClassificationWithTeacher.from_pretrained("facebook/deit-small-distilled-patch16-224").cuda(),
    DeiTForImageClassificationWithTeacher.from_pretrained("facebook/deit-base-distilled-patch16-224").cuda(),
    # MobileViTForImageClassification.from_pretrained("apple/mobilevit-small").cuda(),

]

model_names = [
    'MobNetV2', 'MobNetV3_S', 'MobNetV3_L',
    'VGG11_BN', 'VGG13_BN', 'VGG16_BN', 'VGG19_BN',
    'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
    'ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32',
    'DeiT-tiny(D)', 'DeiT-small(D)', 'DeiT-base(D)',
    # 'MobViT-small',

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

@torch.no_grad()
def validate_tf(test_loader, model, criterion):
    model.eval()
    
    num_data = 0
    num_correct = 0
    sum_loss = 0
    
    pbar = tqdm(test_loader, leave=False, total=len(test_loader))
    for data, target in pbar:
        data, target = data.cuda(), target.cuda()

        output = model(data)
        output = output.logits
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


for mname, model in zip(model_names, model_list):
    print(f"Validating {mname}")

    val_loss, val_acc = validate(val_loader, model, criterion)

    model_vloss.append(val_loss)
    model_vacc.append(val_acc)

    num_params = sum(p.numel() for p in model.parameters())
    model_nparams.append(num_params)

for mname, model in zip(model_names[-4:], model_deit_list):
    print(f"Validating {mname}")

    val_loss, val_acc = validate_tf(val_loader, model, criterion)

    model_vloss.append(val_loss)
    model_vacc.append(val_acc)

    num_params = sum(p.numel() for p in model.parameters())
    model_nparams.append(num_params)


print("+========================================================+")
print("|                 IMAGENET-1K VALIDATION                 |")
print("|--------------------------------------------------------|")
print("|  Model (#PARAM)                 |  V LOSS  |   V ACC   |")
print("|========================================================|")

for mname, vloss, vacc, nparams in zip(model_names, model_vloss, model_vacc, model_nparams):
    print(f'|  {mname:16s} ({nparams:11,d}) |  {vloss:.4f}  |  {vacc*100:.2f} %  |')

print("+========================================================+")
