import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

@torch.no_grad()
def validate_single(epoch, model, test_loader, criterion, bin_op):
    model.eval()
    if bin_op is not None:
        bin_op.binarization()
    
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

        pbar.set_description(f"EPOCH {epoch:3d} | V LOSS: {avg_loss:.4f}, V ACC: {accuracy*100:.4f}%")

    if bin_op is not None:
        bin_op.restore()
    avg_loss = sum_loss / num_data
    accuracy = num_correct / num_data
    
    return avg_loss, accuracy


@torch.no_grad()
def validate_multi_feature(epoch, model_list, test_loader, criterion, bin_op_list):
    for model in model_list:
        model.eval()
    
    if bin_op_list is not None:
        for bin_op in bin_op_list:
            bin_op.binarization()

    num_data = 0
    num_correct = 0
    sum_loss = 0

    pbar = tqdm(enumerate(test_loader), leave=False, total=len(test_loader))
    for batch_idx, (data, target) in pbar:
        data, target = data.cuda(), target.cuda()
        
        output_list = [model(data) for model in model_list]
        output_avg = sum(output_list) / len(output_list)
        
        loss = criterion(output_avg, target)
        
        _, predicted = torch.max(output_avg.data, 1)
        num_data += target.size(0)
        num_correct += (predicted == target).sum().item()
        sum_loss += loss.item() * target.size(0)
    
        avg_loss = sum_loss / num_data
        accuracy = num_correct / num_data

        pbar.set_description(f"EPOCH {epoch:3d} | V LOSS: {avg_loss:.4f}, V ACC: {accuracy*100:.4f}%")

    if bin_op_list is not None:
        for bin_op in bin_op_list:
            bin_op.restore()
    avg_loss = sum_loss / num_data
    accuracy = num_correct / num_data

    return avg_loss, accuracy


@torch.no_grad()
def validate_multi_hardvote(epoch, model_list, test_loader, criterion, bin_op_list):
    for model in model_list:
        model.eval()

    if bin_op_list is not None:
        for bin_op in bin_op_list:
            bin_op.binarization()

    num_data = 0
    num_correct = 0
    sum_loss = 0

    pbar = tqdm(enumerate(test_loader), leave=False, total=len(test_loader))
    for batch_idx, (data, target) in pbar:
        data, target = data.cuda(), target.cuda()

        vote = torch.zeros(target.size(0), 10).cuda()
        for model in model_list:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            for i in range(target.size(0)):
                vote[i][predicted[i]] += 1

        _, predicted = torch.max(vote, 1)
        num_data += target.size(0)
        num_correct += (predicted == target).sum().item()
        sum_loss += criterion(vote, target).item() * target.size(0)

        avg_loss = sum_loss / num_data
        accuracy = num_correct / num_data

        pbar.set_description(f"EPOCH {epoch:3d} | V LOSS: {avg_loss:.4f}, V ACC: {accuracy*100:.4f}%")

    if bin_op_list is not None:
        for bin_op in bin_op_list:
            bin_op.restore()

    avg_loss = sum_loss / num_data
    accuracy = num_correct / num_data

    return avg_loss, accuracy


@torch.no_grad()
def validate_multi_softvote(epoch, model_list, test_loader, criterion, bin_op_list):
    for model in model_list:
        model.eval()

    if bin_op_list is not None:
        for bin_op in bin_op_list:
            bin_op.binarization()

    num_data = 0
    num_correct = 0
    sum_loss = 0

    pbar = tqdm(enumerate(test_loader), leave=False, total=len(test_loader))
    for batch_idx, (data, target) in pbar:
        data, target = data.cuda(), target.cuda()

        vote = torch.zeros(target.size(0), 10).cuda()
        for model in model_list:
            output = model(data)
            vote += F.softmax(output, dim=1)

        _, predicted = torch.max(vote, 1)
        num_data += target.size(0)
        num_correct += (predicted == target).sum().item()
        sum_loss += criterion(vote, target).item() * target.size(0)

        avg_loss = sum_loss / num_data
        accuracy = num_correct / num_data

        pbar.set_description(f"EPOCH {epoch:3d} | V LOSS: {avg_loss:.4f}, V ACC: {accuracy*100:.4f}%")

    if bin_op_list is not None:
        for bin_op in bin_op_list:
            bin_op.restore()

    avg_loss = sum_loss / num_data
    accuracy = num_correct / num_data

    return avg_loss, accuracy
