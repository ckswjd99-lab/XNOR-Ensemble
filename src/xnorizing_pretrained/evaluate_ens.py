import torch
import torch.nn as nn
import torchvision

import timm
import time

from tqdm import tqdm

from binarizer import BinOp
from dataloader import get_imnet1k_dataloader


CHECKPOINT_PATHS = [
    # './saves/fs_resnet18d.ra2_in1k_bw_best.pth',
    # './saves/fs_resnet18d.ra2_in1k_bw_e4_best.pth',
    # './saves/ft_resnet18d_ra2_in1k_wb_best.pth',
    # './saves/ft_resnet18d.ra2_in1k_bw_e6_best.pth',
    # './saves/ft_resnet18d.ra2_in1k_bw_e4toe6_best.pth'
    # './saves/ft_resnet50.a1_in1k_bw_best.pth',
    # './saves/ft_resnet50.a1_in1k_bw_02_best.pth',
    # './saves/ft_resnet50.a1_in1k_bw_03_best.pth',
    # './saves/ft_resnet50.a1_in1k_bw_04_best.pth',
    # './saves/ft_resnet50.a1_in1k_bw_e3toe5_01_best.pth',
    # './saves/ft_resnet50.a1_in1k_bw_e3toe6_best.pth',
    # './saves/ft_resnet50.a1_in1k_bw.pth',
    # './saves/ft_resnet50.a1_in1k_bw_02.pth',
    # './saves/ft_resnet50.a1_in1k_bw_03.pth',
    # './saves/ft_resnet50.a1_in1k_bw_04.pth',
    # './saves/ft_resnet50.a1_in1k_bw_e3toe5_01.pth',
    # './saves/ft_resnet50.a1_in1k_bw_e3toe6.pth',
]
BATCH_SIZE = 128

def softmax_to_onehot(prob_vec):
    _, predicted_class = torch.max(prob_vec, dim=1)
    
    # one-hot 벡터로 변환
    one_hot = torch.zeros(prob_vec.size())
    one_hot = one_hot.cuda()
    one_hot.scatter_(1, predicted_class.view(-1, 1), 1)

    # prob_vec 을 약간 추가
    one_hot = one_hot + 0.01 * prob_vec
    
    return one_hot


@torch.no_grad()
def validate(epoch, model_list, test_loader, criterion):
    for model in model_list:
        model.eval()
    
    num_data = 0
    num_correct = 0
    sum_loss = 0
    sum_confidence = 0  # confidence = estimated probability of the target class

    pbar = tqdm(enumerate(test_loader), leave=False, total=len(test_loader))
    for batch_idx, (data, target) in pbar:

        data, target = data.cuda(), target.cuda()

        # prob_outputs = [nn.Softmax(dim=1)(model(data)) for model in model_list]
        # hard_votes = [softmax_to_onehot(prob_output) for prob_output in prob_outputs]
        outputs = [nn.Softmax(dim=1)(model(data)) for model in model_list]
        # outputs = [model(data) for model in model_list]

        # outputs = [(2 ** max(0, i-1)) * output for i, output in enumerate(outputs)]

        # outputs = hard_votes
        output = sum(outputs) / len(model_list)

        loss = criterion(output, target)
        
        _, predicted = torch.max(output.data, 1)
        num_data += target.size(0)
        num_correct += (predicted == target).sum().item()
        sum_loss += loss.item() * target.size(0)
        sum_confidence += output[torch.arange(target.size(0)), target].sum().item()
    
        avg_loss = sum_loss / num_data
        accuracy = num_correct / num_data
        avg_confidence = sum_confidence / num_data

        pbar.set_description(f"EPOCH {epoch:3d} | V LOSS: {avg_loss:.4f}, V ACC: {accuracy*100:.4f}%, V CONFI: {avg_confidence:.4f}")

    avg_loss = sum_loss / num_data
    accuracy = num_correct / num_data
    avg_confidence = sum_confidence / num_data
    
    return avg_loss, accuracy, avg_confidence


# models = [timm.create_model('resnet18d.ra2_in1k', pretrained=False).cuda() for _ in range(len(CHECKPOINT_PATHS))]
# models = [timm.create_model('resnet50.a1_in1k', pretrained=False).cuda() for _ in range(len(CHECKPOINT_PATHS))]
# models = [torchvision.models.resnet50(pretrained=True).cuda()]
model_names = [
    # 'convmixer_768_32.in1k',                        # 80.03
    # 'convnext_nano.d1h_in1k',                       # 80.60
    #>> 81.83
    # 'dpn107.mx_in1k',                               # 80.13
    # 'ecaresnetlight.miil_in1k',                     # 80.46
    # 'efficientformer_l1.snap_dist_in1k',            # 80.29
    #>> 82.98
    # 'gernet_m.idstcv_in1k',                         # 80.47
    # 'hgnetv2_b2.ssld_stage1_in22k_in1k',            # 80.49
    # 'legacy_seresnext101_32x4d.in1k',               # 80.26
    # 'mixnet_xl.ra_in1k',                            # 80.48
    # 'mobilenetv4_hybrid_medium.e500_r224_in1k',     # 80.38
    # 'poolformerv2_s24.sail_in1k',                   # 80.74
    #>> 83.46
    
    # 'hardcorenas_a.miil_green_in1k',                # 75.96
    # 'densenet169.tv_in1k',                          # 75.90
    # 'convnext_atto_ols.a2_in1k',                    # 75.89 
    # 'mobilenetv3_large_100.ra_in1k',                # 75.77
    # 'mobileone_s1.apple_in1k',                      # 75.66
    # 'tf_mixnet_s.in1k',                             # 75.68
    # 'regnety_004.tv2_in1k',                         # 75.51
    # 'resnest14d.gluon_in1k',                        # 75.63

    # 'resnet34.a2_in1k',                     # 75.55

    # 'rexnet_100.nav_in1k',                  # 77.85, 0.41
    # 'hardcorenas_d.miil_green_in1k',         # 77.45, 0.3
    # 'ghostnetv2_100.in1k',                  # 75.16, 0.18
    # 'repghostnet_111.in1k',                 # 75.07, 0.18
    # 'hardcorenas_a.miil_green_in1k',        # 75.96
    # 'mobilenetv3_large_100.ra_in1k',        # 75.77
    # 'mobilenetv3_rw.rmsp_in1k',             # 75.61
    # 'tf_mixnet_s.in1k',                     # 75.68
    # 'mobilenetv3_large_100.ra_in1k',        # 75.77
    # 'semnasnet_100.rmsp_in1k',              # 75.45
    # 'efficientnet_lite0.ra_in1k',           # 75.50
    # 'fbnetc_100.rmsp_in1k',                 # 75.03
    # 'regnety_004.tv2_in1k',
    # 'mobilenetv2_110d.ra_in1k',

    # 'efficientnet_lite0.ra_in1k',           # 75.50, 0.4
    # 'fbnetc_100.rmsp_in1k',                 # 75.03, 0.4
    # #>> 76.95
    # 'tf_mixnet_m.in1k',                     # 76.96, 0.36
    # #>> 78.52
    # 'repvit_m1.dist_in1k',                  # 78.54, 0.83
    # #>> 79.85
    # 'levit_192.fb_dist_in1k',               # 79.85, 0.66
    # #>> 81.04
    # 'pit_s_224.in1k',                       # 81.13, 2.88
    # #>> 82.04
    # 'convnextv2_nano.fcmae_ft_in22k_in1k',  # 82.04, 2.46
    # #>> 83.23
    # 'tiny_vit_11m_224.dist_in22k_ft_in1k',  # 83.23, 2.04
    # #>> 84.23
    # 'convnext_tiny.in12k_ft_in1k',          # 84.19, 4.47
    # #>> 84.84
    # 'nextvit_small.bd_ssld_6m_in1k',        # 84.86, 5.81
    # #>> 85.36
    # 'caformer_s36.sail_in22k_ft_in1k',      # 85.79, 8.00
    # #>> 86.06

    'efficientvit_l3.r224_in1k',
    'caformer_s36.sail_in22k_ft_in1k',
    'convnextv2_large.fcmae_ft_in1k',
    'deit3_base_patch16_224.fb_in22k_ft_in1k',
]
models = [timm.create_model(mname, pretrained=True).cuda() for mname in model_names]

for mname, model in zip(model_names, models):
    num_params = sum(p.numel() for p in model.parameters())
    print(f"{mname:40s}: {num_params:,d}")

# bin_ops = [BinOp(model) for model in models]

criterion = nn.CrossEntropyLoss().cuda()
# criterion = nn.NLLLoss().cuda()

# for model, checkpoint, bin_op in zip(models, CHECKPOINT_PATHS, bin_ops):
#     checkpoint = torch.load(checkpoint)
#     model.load_state_dict(checkpoint['model'])

#     bin_op.binarization()

train_loader, val_loader = get_imnet1k_dataloader(batch_size=BATCH_SIZE, augmentation=False)

# validate each
# for mname, model in zip(model_names, models):
#     val_loss, val_acc, val_confi = validate(0, [model], val_loader, criterion)

#     print(f"Model {mname}: V LOSS {val_loss:.4f}, V ACC {val_acc*100:.2f}, V CONFI {val_confi:.4f}")

# validate ensemble
val_loss, val_acc, val_confi = validate(0, models, val_loader, criterion)

print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}, Validation confidence: {val_confi:.4f}")

# Model efficientvit_l3.r224_in1k: V LOSS 6.0946, V ACC 85.40, V CONFI 0.8148
# Model caformer_s36.sail_in22k_ft_in1k: V LOSS 6.1743, V ACC 85.59, V CONFI 0.7350
# Model convnextv2_large.fcmae_ft_in1k: V LOSS 6.1629, V ACC 85.76, V CONFI 0.7463
# Model deit3_base_patch16_224.fb_in22k_ft_in1k: V LOSS 6.1632, V ACC 85.49, V CONFI 0.7461
# Validation loss: 6.1487, Validation accuracy: 0.8710, Validation confidence: 0.7605