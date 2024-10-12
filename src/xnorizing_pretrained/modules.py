import torch
import torch.nn as nn

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
            input = input.sign()
        
        return input, mean

    @staticmethod
    def backward(ctx, grad_output, grad_output_mean):
        input, activate = ctx.saved_tensors
        grad_input = grad_output.clone()
        if activate:
            grad_input[input.ge(1)] = 0
            grad_input[input.le(-1)] = 0
        
        return grad_input, None


class Conv2dXNOR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
        super(Conv2dXNOR, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.bn = nn.BatchNorm2d(in_channels, momentum=0.9)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        self.scaler = 1
        self.bn.weight.data = self.bn.weight.data.zero_().add(1.0)

    def init_from_conv2d(self, conv2d_module):
        mean = conv2d_module.weight.data.abs().mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        self.weight.data.copy_(conv2d_module.weight.data / self.scaler)

    def forward(self, x):
        x = self.bn(x)
        x, _ = BinActive.apply(x)

        return nn.functional.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        ) * self.scaler