import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2d(nn.Conv2d):
    ##################
    ### Problem 2(a): Implement MaskedConv2d
    def __init__(self, **args):
        type_ = args['type_']
        args.pop('type_')
        super().__init__(**args)
        self.type_ = type_
    def forward(self, x):
        kernel_size = self.kernel_size
        if not isinstance(kernel_size, int): kernel_size = kernel_size[0]
        self.weight.data[...,kernel_size//2+1:,:].zero_()
        self.weight.data[...,kernel_size//2,kernel_size//2+1:].zero_()
        if self.type_ == 'A':
            self.weight.data[...,kernel_size//2,kernel_size//2].zero_()
        return super().forward(x)
    ##################

class PixelCNNDiscrete(nn.Module):
    ##################
    ### Problem 2(b): Implement PixelCNN
    def __init__(self,classes=256):
        super().__init__()
        self.layers = nn.Sequential(
            MaskedConv2d(in_channels=1,out_channels=64,kernel_size=11,padding=5,type_='A'),
            nn.ReLU(),
            MaskedConv2d(in_channels=64,out_channels=64,kernel_size=11,padding=5,type_='B'),
            nn.ReLU(),
            MaskedConv2d(in_channels=64,out_channels=64,kernel_size=11,padding=5,type_='B'),
            nn.ReLU(),
            MaskedConv2d(in_channels=64,out_channels=64,kernel_size=11,padding=5,type_='B'),
            nn.ReLU(),
            MaskedConv2d(in_channels=64,out_channels=128,kernel_size=11,padding=5,type_='B'),
            nn.ReLU(),
            nn.Conv2d(128,classes,kernel_size=1)
        )
        self.classes = classes
        
    def forward(self, x):
        return self.layers(x)
    ##################
    
class PixelCNNContinous(nn.Module):
    ##################
    ### Problem 2(b): Implement PixelCNN
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            MaskedConv2d(in_channels=1,out_channels=64,kernel_size=11,padding=5,type_='A'),
            nn.ReLU(),
            MaskedConv2d(in_channels=64,out_channels=64,kernel_size=11,padding=5,type_='B'),
            nn.ReLU(),
            MaskedConv2d(in_channels=64,out_channels=64,kernel_size=11,padding=5,type_='B'),
            nn.ReLU(),
            MaskedConv2d(in_channels=64,out_channels=64,kernel_size=11,padding=5,type_='B'),
            nn.ReLU(),
            MaskedConv2d(in_channels=64,out_channels=128,kernel_size=11,padding=5,type_='B'),
            nn.ReLU(),
            nn.Conv2d(128,1,kernel_size=1) # predicting mu and logvar doesn't work
        )
        
    def forward(self, x):
        return self.layers(x)
        # return self.layers(x).chunk(2,dim=1)
    ##################