import torch
import torch.nn as nn
import torch.nn.functional as F

import time
from tqdm import tqdm
import os

from binarizer import BinOp
from dataloader import get_CIFAR10_dataset
from resnet_custom import ResNet9
from trainer import validate_single, validate_multi_feature

torch.autograd.set_detect_anomaly(True)

BATCH_SIZE = 128
EPOCHS = 20
LR_START = 1e-3
LR_END = 1e-5
SOFTENING = 0.0

TEACHER_CKPT = "saves/resnet9_epoch_75.pth"
PREFIX = "resnet9_bw_dist_naive"
COLLEAGUES_CKPTS = ['saves/' + f for f in os.listdir('saves') if f.startswith(PREFIX) and f.endswith("_final.pth")]
CURRENT_CKPT = f"saves/{PREFIX}_{len(COLLEAGUES_CKPTS):02d}.pth"

print(f"Teacher: {TEACHER_CKPT}")
print(f"Current: {CURRENT_CKPT}")

VALIDATE_EACH = True
MODEL_ARCH = ResNet9


train_loader, val_loader = get_CIFAR10_dataset(root='../data', batch_size=BATCH_SIZE)

criterion_xent = nn.CrossEntropyLoss().cuda()

# LOAD TEACHER
teacher = MODEL_ARCH().cuda()
teacher_ckpt = torch.load(TEACHER_CKPT, map_location='cuda', weights_only=False)['model_state_dict']
teacher.load_state_dict(teacher_ckpt)

if VALIDATE_EACH:
    teacher_vloss, teacher_vacc = validate_single(0, teacher, val_loader, criterion_xent, None)

    print(f"Teacher: V LOSS: {teacher_vloss:.4f}, V ACC: {teacher_vacc*100:.2f}%")


# TRAIN NEW STUDENT
student = MODEL_ARCH().cuda()
student.load_state_dict(teacher_ckpt)
student_binop = BinOp(student)

training_params_dict = dict(student.named_parameters())
training_params_names = [name for name, param in training_params_dict.items() if 'conv.0.' not in name and 'fc.' not in name]
training_params = [training_params_dict[name] for name in training_params_names]
print(f"Training params: {len(training_params)}")
print(training_params_names)

optimizer = torch.optim.Adam(training_params, lr=LR_START)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(LR_END/LR_START)**(1/(EPOCHS-1)))


best_vacc = 0
for epoch in range(EPOCHS):
    start_time = time.time()

    # TRAINING

    student.train()

    s_tloss_sum = 0
    s_num_data = 0
    s_num_correct = 0

    pbar = tqdm(train_loader, total=len(train_loader), leave=False)
    for input, target in pbar:
        input, target = input.cuda(), target.cuda()

        optimizer.zero_grad()
        student_binop.binarization()

        teacher_output = teacher(input)

        student_output = student(input)

        loss = criterion_xent(student_output, target)

        loss.backward()

        student_binop.restore()
        student_binop.updateBinaryGradWeight()

        optimizer.step()

        _, predicted = torch.max(student_output.data, 1)
        s_num_data += target.size(0)
        s_num_correct += (predicted == target).sum().item()
        s_tloss_sum += loss.item() * target.size(0)

        s_tloss_avg = s_tloss_sum / s_num_data
        s_tacc_avg = s_num_correct / s_num_data

        pbar.set_description(f"EPOCH {epoch:3d} | T LOSS: {s_tloss_avg:.4f}, T ACC: {s_tacc_avg*100:.2f}%")

    # VALIDATION
    
    s_vloss_sum = 0
    s_num_data = 0
    s_num_correct = 0

    student.eval()
    student_binop.binarization()

    pbar = tqdm(val_loader, total=len(val_loader), leave=False)
    for input, target in pbar:
        input, target = input.cuda(), target.cuda()

        student_output = student(input)

        loss = criterion_xent(student_output, target)

        _, predicted = torch.max(student_output.data, 1)
        s_num_data += target.size(0)
        s_num_correct += (predicted == target).sum().item()
        s_vloss_sum += loss.item() * target.size(0)

        student_vloss = s_vloss_sum / s_num_data
        student_vacc = s_num_correct / s_num_data

        pbar.set_description(f"EPOCH {epoch:3d} | V LOSS: {student_vloss:.4f}, V ACC: {student_vacc*100:.2f}%")

    student_binop.restore()


    end_time = time.time()

    is_best = student_vacc > best_vacc
    if is_best:
        best_vacc = student_vacc
        torch.save(student.state_dict(), CURRENT_CKPT.replace('.pth', '_best.pth'))

    print(f"EPOCH {epoch:3d}, LR: {scheduler.get_last_lr()[0]:.4e} | T LOSS: {s_tloss_avg:.4f}, T ACC: {s_tacc_avg*100:.2f}%, V LOSS: {student_vloss:.4f}, V ACC: {student_vacc*100:.2f}% | ETA: {int(end_time-start_time):8,d} sec")

    scheduler.step()

os.rename(CURRENT_CKPT.replace('.pth', '_best.pth'), CURRENT_CKPT.replace('.pth', '_final.pth'))