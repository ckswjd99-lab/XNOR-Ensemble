import torch

from tqdm import tqdm

@torch.enable_grad()
def train(epoch, model, train_loader, optimizer, criterion, bin_op):
    model.train()
    
    num_data = 0
    num_correct = 0
    sum_loss = 0

    lr_epoch = optimizer.param_groups[0]['lr']
    
    pbar = tqdm(enumerate(train_loader), leave=False, total=len(train_loader))
    for batch_idx, (data, target) in pbar:
        # process the weights including binarization
        optimizer.zero_grad()
        bin_op.binarization()
        
        # forwarding
        data, target = data.cuda(), target.cuda()
        output = model(data)
        
        # backwarding
        loss = criterion(output, target)
        loss.backward(retain_graph=True)
        
        # restore weights
        bin_op.restore()
        bin_op.updateBinaryGradWeight()
        
        optimizer.step()
        
        _, predicted = torch.max(output.data, 1)
        num_data += target.size(0)
        num_correct += (predicted == target).sum().item()
        sum_loss += loss.item() * target.size(0)
    
        avg_loss = sum_loss / num_data
        accuracy = num_correct / num_data

        pbar.set_description(f"EPOCH {epoch:3d}, LR {lr_epoch:.4e} | T LOSS: {avg_loss:.4f}, T ACC: {accuracy*100:.2f}%")
    
    avg_loss = sum_loss / num_data
    accuracy = num_correct / num_data

    return avg_loss, accuracy

@torch.no_grad()
def validate(epoch, model, test_loader, criterion, bin_op):
    model.eval()
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

    bin_op.restore()
    avg_loss = sum_loss / num_data
    accuracy = num_correct / num_data
    
    return avg_loss, accuracy