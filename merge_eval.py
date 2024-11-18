import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm,trange
import torchvision.utils as vutils
import os

### Schedules ###
T = 1000
angles = torch.linspace(0.1, torch.pi-0.1, T)
alpha_bars = (1 + torch.cos(angles)) / 2
alphas = torch.ones_like(alpha_bars,dtype=torch.float)
alphas[1:] = alpha_bars[1:] / alpha_bars[:-1]
sigmas = torch.sqrt(1-alphas)

### Models ###
from vanilla_unet import UNet as ModelCls
from pixelcnn import PixelCNNDiscrete, PixelCNNContinous

@torch.no_grad()
def sample_pixelcnn_discrete(model, num=64,ep=0):
    x = torch.zeros(num, 1, 28, 28).cuda()
    for i in trange(28,desc=f'Epoch {ep} sampling'):
        for j in range(28):
            logits = model(x)
            probs = F.softmax(logits[:, :, i, j],dim=1)
            x[:, :, i, j] = (torch.multinomial(probs, 1).float()) / model.classes
    # grid = vutils.make_grid(x, nrow=8)
    # vutils.save_image(grid, f'samples/{ep}_pixelcnn.png')
    return x

@torch.no_grad()
def sample_pixelcnn_continous(model, num=64,ep=0):
    x = torch.zeros(num, 1, 28, 28).cuda()
    for i in trange(28,desc=f'Epoch {ep} sampling'):
        for j in range(28):
            mu = model(x); logvar = torch.tensor(-2.0).cuda().repeat(*x.shape)
            # mu, logvar = model(x)
            x[:, :, i, j] = torch.normal(mu[:, :, i, j], torch.exp(logvar[:, :, i, j]/2)).clamp(0,1)
    grid = vutils.make_grid(x, nrow=8)
    vutils.save_image(grid, f'samples/{ep}_pixelcnn_c.png')
    return x

@torch.no_grad()
def sample_halfddpm(model,data,num=64,ep=0):
    x = data * torch.sqrt(alpha_bars[T//2]) + torch.randn_like(data) * torch.sqrt(1-alpha_bars[T//2])
    grid = vutils.make_grid(x, nrow=8)
    vutils.save_image(grid, f'samples/{ep}_init.png')
    
    for t in trange(T//2-1,-1,-1,desc=f'Epoch {ep} sampling'):
        w1 = 1/torch.sqrt(alphas[t]).cuda()
        w2 = (1-alphas[t])/torch.sqrt(1-alpha_bars[t]).cuda()
        x = w1 * (x - w2 * model(x,torch.tensor([t]).cuda().repeat(x.shape[0],))) + sigmas[t].cuda().reshape(-1,1,1,1) * torch.randn_like(x)
    # make grid
    # grid = vutils.make_grid(x, nrow=8)
    # vutils.save_image(grid, f'samples/{ep}_sample.png')
    return x

def evaluate(pixelcnn_path, model_path):
    pixelcnn = PixelCNNContinous().cuda()
    # pixelcnn = PixelCNNDiscrete().cuda()
    pixelcnn.load_state_dict(torch.load(f'checkpoints/{pixelcnn_path}'))
    
    model = ModelCls().cuda()
    model.load_state_dict(torch.load(f'checkpoints/{model_path}'))
    
    # sample from pixelcnn
    # x = torch.randn(64,1,28,28).cuda()
    x = sample_pixelcnn_discrete(pixelcnn) if isinstance(pixelcnn,PixelCNNDiscrete) else sample_pixelcnn_continous(pixelcnn)
    grid = vutils.make_grid(x, nrow=8)
    vutils.save_image(grid, f'samples/merge_just_{pixelcnn_path}.png')
    # sample from halfddpm
    x = sample_halfddpm(model,x)
    grid = vutils.make_grid(x, nrow=8)
    vutils.save_image(grid, f'samples/merge_{pixelcnn_path}_and_{model_path}.png')
    
if __name__ == '__main__':
    evaluate('9_pixelcnn_c.pth','49.pth')
    # evaluate('Gaussian','49.pth')