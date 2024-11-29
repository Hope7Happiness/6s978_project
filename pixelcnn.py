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
    
class MaskedConv2d_label(nn.Conv2d):
    ##################
    ### Problem 2(a): Implement MaskedConv2d
    def __init__(self, **args):
        type_ = args['type_']
        args.pop('type_')
        super().__init__(**args)
        self.type_ = type_
        self.cls_cond = nn.Embedding(10, self.out_channels * 2)
        self.cls_cond.weight.data.zero_()
    def forward(self, x, label):
        kernel_size = self.kernel_size
        if not isinstance(kernel_size, int): kernel_size = kernel_size[0]
        self.weight.data[...,kernel_size//2+1:,:].zero_()
        self.weight.data[...,kernel_size//2,kernel_size//2+1:].zero_()
        if self.type_ == 'A':
            self.weight.data[...,kernel_size//2,kernel_size//2].zero_()
        # print('self.weight:',self.weight.data)
        x = super().forward(x)
        # return x
        cls_scale, cls_shift = self.cls_cond(label).chunk(2,dim=1)
        return x * (1 + cls_scale.unsqueeze(-1).unsqueeze(-1)) + cls_shift.unsqueeze(-1).unsqueeze(-1)
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
            nn.Conv2d(128,32,kernel_size=1), # predicting mu and logvar doesn't work
            nn.ReLU(),
            nn.Conv2d(32,2,kernel_size=1) # predicting mu and logvar doesn't work
        )
        self.layers[-1].weight.data.zero_()
        self.layers[-1].bias.data.zero_()
        
    def forward(self, x):
        # return self.layers(x)
        mu, logvar = self.layers(x).chunk(2,dim=1)
        # avoid cheating by very small logvar
        logvar = torch.clamp(logvar,-5,5)
        return mu, logvar
    ##################
    
class t_relu(nn.Module):
    def forward(self,x,label):
        return x.relu()
class PixelCNNContinous_cond(nn.Module):
    ##################
    ### Problem 2(b): Implement PixelCNN
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            MaskedConv2d_label(in_channels=1,out_channels=64,kernel_size=11,padding=5,type_='A'),
            t_relu(),
            MaskedConv2d_label(in_channels=64,out_channels=64,kernel_size=11,padding=5,type_='B'),
            t_relu(),
            MaskedConv2d_label(in_channels=64,out_channels=64,kernel_size=11,padding=5,type_='B'),
            t_relu(),
            MaskedConv2d_label(in_channels=64,out_channels=64,kernel_size=11,padding=5,type_='B'),
            t_relu(),
            MaskedConv2d_label(in_channels=64,out_channels=64,kernel_size=11,padding=5,type_='B'),
        )
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(64,2,kernel_size=1), # predicting mu and logvar doesn't work
            # nn.ReLU(),
            # nn.Conv2d(32,2,kernel_size=1) # predicting mu and logvar doesn't work
        )
        # self.layers[-1].weight.data.zero_()
        # self.layers[-1].bias.data.zero_()
        
    def forward(self, x, label):
        # return self.layers(x)
        for i,l in enumerate(self.layers):
            # print('l is:',type(l)) 
            if i == 0:
                x = l(x,label)
            else:
                x = l(x,label) + x
        mu, logvar = self.head(x).chunk(2,dim=1)
        # with torch.no_grad():
        #     if torch.rand(1).item() > 0.99:
        #         print('mu:',mu[0])
        #         print('logvar:',logvar[0])
        # avoid cheating by very small logvar
        # logvar = torch.clamp(logvar,-5,5)
        return mu, logvar
    ##################