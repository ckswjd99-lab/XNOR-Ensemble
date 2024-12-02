import torch.nn as nn
import numpy
import torch

from modules import Conv2dXNOR

class BinOp():
    def __init__(self, model):
        # count the number of Conv2d
        count_Conv2d = 0
        for m in model.modules():
            if isinstance(m, Conv2dXNOR) or isinstance(m, nn.Conv2d):
                count_Conv2d = count_Conv2d + 1

        start_range = 1
        end_range = count_Conv2d-2
        self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        index = -1
        for m in model.modules():
            if isinstance(m, Conv2dXNOR) or isinstance(m, nn.Conv2d):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)
        
        self.model = model

    def binarization(self):
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()
    
    def quantization(self, num_bits):
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.quantizeConvParams(num_bits)

    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            s = self.target_modules[index].data.size()
            negMean = self.target_modules[index].data.mean(1, keepdim=True).\
                    mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = \
                    self.target_modules[index].data.clamp(-1.0, 1.0)            

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarizeConvParams(self):
        for index in range(self.num_of_params):
            n = self.target_modules[index].data[0].nelement()
            s = self.target_modules[index].data.size()
            m = self.target_modules[index].data.norm(1, 3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n)

            self.target_modules[index].data = \
                    self.target_modules[index].data.sign().mul(m.expand(s))

    def quantizeConvParams(self, num_bits):
        for index in range(self.num_of_params):
            # abs mean of each output channel
            output_channel_mean = self.target_modules[index].data.abs()\
                    .mean(3, keepdim=True).mean(2, keepdim=True).mean(1, keepdim=True)\
                    .expand_as(self.target_modules[index].data)
            
            # normalize and clamp
            normalized_weight = self.target_modules[index].data.div(output_channel_mean).clamp_(-1.0, 1.0)
            
            # quantize
            quantized_weight = torch.round(normalized_weight.mul(2**(num_bits-1))).div(2**(num_bits-1))

            self.target_modules[index].data = quantized_weight

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            m = weight.norm(1, 3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            
            m[weight.lt(-1.0)].fill_(0)
            m[weight.gt(1.0)].fill_(0)
            # m = m.add(1.0/n).mul(1.0-1.0/s[1]).mul(n)
            # self.target_modules[index].grad.data = \
            #         self.target_modules[index].grad.data.mul(m)
            m = m.mul(self.target_modules[index].grad.data)
            m_add = weight.sign().mul(self.target_modules[index].grad.data)
            m_add = m_add.sum(3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            m_add = m_add.mul(weight.sign())
            self.target_modules[index].grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(n)

    def countBinaryParams(self):
        num_binary_params = 0
        for index in range(self.num_of_params):
            num_binary_params += self.target_modules[index].data.nelement()
        return num_binary_params