import torch.nn as nn
import torch
import torch.nn.functional as F


class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''

    @staticmethod
    def forward(ctx, input, activate=True):
        ctx.save_for_backward(input, torch.tensor(activate))
        size = input.size()
        mean = torch.mean(input.abs(), 1, keepdim=True)
        if activate:
            input = input.clamp(-1.0, 1.0)
            # input = input.sign()
        
        return input, mean

    @staticmethod
    def backward(ctx, grad_output, grad_output_mean):
        input, activate = ctx.saved_tensors
        grad_input = grad_output.clone()
        if activate:
            grad_input[input.ge(1)] = 0
            grad_input[input.le(-1)] = 0
        
        return grad_input, None

class BinConv2d(nn.Module):
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, dropout=0, bin_active=True):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout
        self.bin_active = bin_active

        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding)
        
        self.bn = nn.BatchNorm2d(output_channels, momentum=0.9)

        self.bn.weight.data = self.bn.weight.data.zero_().add(1.0)
    
    def forward(self, x):
        x, mean = BinActive.apply(x, self.bin_active)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        x = self.conv(x)
        x = self.bn(x)
        return x

class ResidualBlock(nn.Module):
    """
    A residual block as defined by He et al.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, bin_active=True):
        super(ResidualBlock, self).__init__()
        self.conv_res1 = BinConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bin_active=bin_active)
        self.conv_res2 = BinConv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=1, bin_active=bin_active)

        if stride != 1:
            # in case stride is not set to 1, we need to downsample the residual so that
            # the dimensions are the same when we add them together
            self.downsample = nn.Sequential(
                BinConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False, bin_active=bin_active),
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=False)


    def forward(self, x):
        residual = x

        out = self.relu(self.conv_res1(x))
        out = self.conv_res2(out)
        
        if self.downsample is not None:
            residual = self.downsample(residual)

        # out = self.relu(out)
        out = out + residual
        return out


class ResNet9(nn.Module):
    """
    A Residual network.
    """
    def __init__(self, bin_active=True):
        super(ResNet9, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace=True),
            BinConv2d(64, 128, kernel_size=3, stride=1, padding=1, bin_active=bin_active),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            BinConv2d(128, 256, kernel_size=3, stride=1, padding=1, bin_active=bin_active),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BinConv2d(256, 256, kernel_size=3, stride=1, padding=1, bin_active=bin_active),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Linear(in_features=1024, out_features=10, bias=True)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
        out = self.fc(out)
        return out