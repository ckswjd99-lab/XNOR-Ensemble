import torch
import torch.nn as nn

from modules import Conv2dXNOR

def find_conv2d_keys(model):
    target_modules = []
    for mname, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            target_modules.append(mname)
    return target_modules

def find_parent_module(model, target_mname):
    keys = target_mname.split('.')
    parent_module = model
    for key in keys[:-1]:
        parent_module = getattr(parent_module, key)

    child_key = keys[-1]

    return parent_module, child_key

def create_xnor_conv2d(conv2d_module):
    xnor_conv2d = Conv2dXNOR(
        in_channels=conv2d_module.in_channels,
        out_channels=conv2d_module.out_channels,
        kernel_size=conv2d_module.kernel_size,
        stride=conv2d_module.stride,
        padding=conv2d_module.padding,
        dilation=conv2d_module.dilation,
        groups=conv2d_module.groups,
        bias=conv2d_module.bias
    )
    
    return xnor_conv2d

def xnorize_conv2d(model):
    target_modules = find_conv2d_keys(model)
    for target_module in target_modules:
        parent_module, child_key = find_parent_module(model, target_module)
        new_conv2d = create_xnor_conv2d(getattr(parent_module, child_key))
        new_conv2d.init_from_conv2d(getattr(parent_module, child_key))
        setattr(parent_module, child_key, new_conv2d)
        
    return model
