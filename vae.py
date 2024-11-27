import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.latent_dim = 4
        self.enc = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2 * self.latent_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(self.latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
        )
        
    def encode(self, x):
        x = x.reshape(-1, 784)
        return self.enc(x).chunk(2, dim=1)
    
    def decode(self, z):
        return self.dec(z).reshape(-1, 1, 28, 28)
    
    @staticmethod
    def kl(mu,logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1).mean(dim=0)
    
    def forward(self,x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(logvar/2)
        x_recon = self.decode(z)
        recon_loss = F.mse_loss(x_recon, x) * (x.numel() / x.shape[0])
        kl_loss = self.kl(mu, logvar)
        return recon_loss + kl_loss, recon_loss, kl_loss
    