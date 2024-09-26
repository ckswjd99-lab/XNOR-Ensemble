from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import torch
import argparse
import dataloader
from binarizer import BinOp
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from dataloader import get_CIFAR10_dataset

from tqdm import tqdm

from models import ResNet9


EPOCH = 200
BATCH_SIZE = 128
LR_START = 1e-3
LR_END = 1e-6
ARCH = 'resnet9'
NUM_CLASSES = 10

FORMER_AGENTS = [
    './saves/agents/resnet9_agent0_vacc8706.pth',
]


def init_dataset_dweight(data_loader):
    dweights = torch.ones(len(data_loader.dataset)) / len(data_loader.dataset)
    dweights = dweights.cuda()
    print(dweights.shape)
    return dweights

@torch.no_grad()
def rate_agent(model, binarizer, train_loader, dweights):
    is_correct = torch.zeros(len(train_loader.dataset)).cuda()
    
    model.eval()
    binarizer.binarization()

    num_data = 0
    num_correct = 0
    sum_loss = 0

    pbar = tqdm(enumerate(train_loader), leave=False, total=len(train_loader))
    for i, (data, target) in pbar:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        probability = torch.nn.functional.softmax(output, dim=1)
        predicted = torch.argmax(probability, dim=1)
        prob_of_target = probability[torch.arange(len(target)), target]

        is_correct[i*BATCH_SIZE:min((i+1)*BATCH_SIZE, len(train_loader.dataset))] = (predicted == target).float()
        # is_correct[i*BATCH_SIZE:min((i+1)*BATCH_SIZE, len(train_loader.dataset))] = prob_of_target

        num_data += target.size(0)
        num_correct += (predicted == target).sum().item()
        sum_loss += torch.nn.functional.cross_entropy(output, target, reduce=True) * target.size(0)


    binarizer.restore()

    error = torch.dot((1 - is_correct), dweights) / torch.sum(dweights)
    stage = torch.log((1 - error) / error) + torch.log(torch.Tensor([NUM_CLASSES - 1])).cuda()
    new_dweights = dweights * torch.exp(stage * (1 - is_correct))
    new_dweights = new_dweights / torch.sum(new_dweights)

    avg_loss = sum_loss / num_data
    accuracy = num_correct / num_data

    return stage, new_dweights, avg_loss, accuracy


def train(epoch, model, binarizer, train_loader, dweights, optimizer, criterion):
    model.train()

    num_data = 0
    num_correct = 0
    sum_loss = 0

    lr_epoch = optimizer.param_groups[0]['lr']

    pbar = tqdm(enumerate(train_loader), leave=False, total=len(train_loader))
    for batch_idx, (data, target) in pbar:
        # process the weights including binarization
        binarizer.binarization()

        # forwarding
        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)

        # backwarding
        loss = criterion(output, target)
        loss.backward()

        # restore weights
        binarizer.restore()
        binarizer.updateBinaryGradWeight()

        optimizer.step()

        _, predicted = torch.max(output.data, 1)
        num_data += target.size(0)
        num_correct += (predicted == target).sum().item()
        sum_loss += loss.item() * target.size(0)

        avg_loss = sum_loss / num_data
        accuracy = num_correct / num_data

        pbar.set_description(f"EPOCH {epoch:3d}, LR {lr_epoch:.4e} | T LOSS: {avg_loss:.4f}, T ACC: {accuracy*100:.2f}%")

    return avg_loss, accuracy


@torch.no_grad()
def validate(epoch, model, test_loader, criterion, binarizer):
    model.eval()
    binarizer.binarization()

    num_data = 0
    num_correct = 0
    sum_loss = 0

    pbar = tqdm(enumerate(test_loader), leave=False, total=len(test_loader))
    for batch_idx, (data, target) in pbar:
        data, target = data.cuda(), target.cuda()

        output = model(data)
        loss = criterion(output, target)

        _, predicted = torch.max(output.data, 1)
        num_data += target.size(0)
        num_correct += (predicted == target).sum().item()
        sum_loss += loss.item() * target.size(0)

    avg_loss = sum_loss / num_data
    accuracy = num_correct / num_data

    binarizer.restore()

    return avg_loss, accuracy


@torch.no_grad()
def validate_ensembled(epoch, agents, stage_agents, binarizer_agents, model, stage_model, binarizer_model, test_loader, criterion):
    for agent, binarizer in zip(agents, binarizer_agents):
        agent.eval()
        binarizer.binarization()
    
    model.eval()
    binarizer_model.binarization()

    num_data = 0
    num_correct = 0

    pbar = tqdm(enumerate(test_loader), leave=False, total=len(test_loader))
    for batch_idx, (data, target) in pbar:
        data, target = data.cuda(), target.cuda()
        prediction = torch.zeros(len(data), NUM_CLASSES).cuda()

        for agent, stage in zip(agents, stage_agents):
            output = agent(data)
            prediction += torch.nn.functional.softmax(output, dim=1) * stage

        output = model(data)
        prediction += torch.nn.functional.softmax(output, dim=1) * stage_model

        _, predicted = torch.max(prediction.data, 1)
        num_data += target.size(0)
        num_correct += (predicted == target).sum().item()

    accuracy = num_correct / num_data

    for binarizer in binarizer_agents:
        binarizer.restore()
    binarizer_model.restore()

    return accuracy


if __name__=='__main__':
    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # rate the former agents
    train_loader, test_loader = get_CIFAR10_dataset(root='./data/', batch_size=BATCH_SIZE)
    train_dweights = init_dataset_dweight(train_loader)

    model_agents = [ResNet9().cuda() for _ in range(len(FORMER_AGENTS))]
    binarizers = [BinOp(model) for model in model_agents]
    stages = []
    rated_agents = []

    for idx in range(len(model_agents)):
        model = model_agents[idx]
        binarizer = binarizers[idx]

        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model.load_state_dict(torch.load(FORMER_AGENTS[idx], weights_only=False))

        stage, train_dweights, agent_t_loss, agent_t_acc = rate_agent(model, binarizer, train_loader, train_dweights)
        agent_v_loss, agent_v_acc = validate(0, model, test_loader, nn.CrossEntropyLoss().cuda(), binarizer)
        stages.append(stage)
        rated_agents.append((agent_t_loss, agent_t_acc, agent_v_loss, agent_v_acc))

    print("Stages of former agents")
    for idx, stage in enumerate(stages):
        print(f"    Agent {idx}: STAGE {stage.item():.4f}, T LOSS {rated_agents[idx][0]:.4f}, T ACC {rated_agents[idx][1]*100:.2f}%, V LOSS {rated_agents[idx][2]:.4f}, V ACC {rated_agents[idx][3]*100:.2f}%")

    sampler = torch.utils.data.sampler.WeightedRandomSampler(train_dweights, len(train_dweights))
    train_loader_weighted, _ = get_CIFAR10_dataset(root='./data/', batch_size=BATCH_SIZE, sampler=sampler)

    new_model = ResNet9().cuda()
    new_model = torch.nn.DataParallel(new_model, device_ids=range(torch.cuda.device_count()))
    new_model.load_state_dict(torch.load('./saves/agents/resnet9_agent0_vacc8706.pth', weights_only=False))

    new_binarizer = BinOp(new_model)
    new_optimizer = optim.Adam(new_model.parameters(), lr=LR_START, weight_decay=0.00001)
    criterion = nn.CrossEntropyLoss().cuda()
    new_scheduler = optim.lr_scheduler.ExponentialLR(new_optimizer, gamma=(LR_END/LR_START)**(1/EPOCH))

    best_acc = 0

    for epoch in range(EPOCH):
        # train the new agent
        train_loss, train_acc = train(epoch, new_model, new_binarizer, train_loader_weighted, train_dweights, new_optimizer, criterion)
        val_loss, val_acc = validate(epoch, new_model, test_loader, criterion, new_binarizer)

        new_stage, _, _, _ = rate_agent(new_model, new_binarizer, train_loader, train_dweights)

        ensembled_acc = validate_ensembled(
            epoch, model_agents, stages, binarizers, new_model, new_stage, new_binarizer, test_loader, criterion
        )

        lr_epoch = new_optimizer.param_groups[0]['lr']

        print(f"EPOCH {epoch:3d}, LR {lr_epoch:.4e} | STAGE: {new_stage.item():.4f}, T LOSS: {train_loss:.4f}, T ACC: {train_acc*100:.2f}%, V LOSS: {val_loss:.4f}, V ACC: {val_acc*100:.2f}% | ENS V ACC: {ensembled_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(new_model.state_dict(), './saves/resnet9_best_agent.pth')

        new_scheduler.step()
