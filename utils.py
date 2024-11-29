import torch
from torchvision import datasets, transforms
import torchvision.utils as vutils

tensor_transform = transforms.Compose([
    transforms.ToTensor()
])

import os
try_data_dirs = ['/home/zhh/data','/home/zhh24/data']
for data_dir in try_data_dirs:
    if os.path.exists(data_dir):
        break
assert os.path.exists(data_dir), 'data_dir does not exist'

batch_size = 512
train_dataset = datasets.MNIST(root = data_dir,
									train = True,
									download = True,
									transform = tensor_transform)
test_dataset = datasets.MNIST(root = data_dir,
									train = False,
									download = True,
									transform = tensor_transform)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
							   batch_size = batch_size,
								 shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
							   batch_size = batch_size,
								 shuffle = False)

class Avger(list):
    def __str__(self):
        return f'{sum(self) / len(self):.4f}' if len(self) > 0 else 'N/A'
    
class BatchPrepareBase:
    def process(self, x):
        raise NotImplementedError
    
class SamplerBase:
    def __init__(self):
        os.makedirs('samples', exist_ok=True)
        os.makedirs('checkpoints', exist_ok=True)
        
    def calc(self, model, num):
        raise NotADirectoryError()
    
    @torch.no_grad()
    def sample(self, model, num, desc):
        x = self.calc(model, num)
        grid = vutils.make_grid(x, nrow=8)
        vutils.save_image(grid, f'samples/{desc}.png')
        torch.save(model.state_dict(), f'checkpoints/{desc}.pth')
        
class TrainerBase:
    def __init__(self, model, epochs,lr,desc,preparer:BatchPrepareBase,sampler:SamplerBase, sample_ep=4):
        self.model = model
        self.epochs = epochs
        self.sample_ep = sample_ep
        self.lr = lr
        self.preparer = preparer
        self.sampler = sampler
        self.desc = desc
        
    def run(self):
        raise NotImplementedError