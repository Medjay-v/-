import torch
import torch.nn as nn
import torch.nn.functional as F
from util import plot_loss,instantiate_from_config
from torch.utils.data import DataLoader
from dataset.dataset import LatentDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = instantiate_from_config('./configs/config.yaml')

class VAE(nn.Module):
    def __init__(self, input_channels, latent_dim,output_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, stride=2, padding=1),  #(64,16,16)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),             #(128,8,8)
            nn.ReLU(),
        )

        self.fc_mu = nn.Conv2d(256, latent_dim, kernel_size = 1)
        self.fc_logvar = nn.Conv2d(256, latent_dim, kernel_size = 1)

        self.decoder_fc = nn.Conv2d(latent_dim, 256, kernel_size = 3, padding=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 编码
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)

        # 重参数化
        z = self.reparameterize(mu, log_var)

        # 解码
        h = self.decoder_fc(z)
        x_recon = self.decoder(h)
        return x_recon, mu, log_var

def vae_loss(x_recon, x, mu, log_var, β = 0.6):
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + β * kl_loss,recon_loss,kl_loss



def vae_train():
    dataset = LatentDataset(config.hr_path)
    hr_train_tensor = dataset.hr_tensorDataset
    train_loader = DataLoader(
        hr_train_tensor,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    vae = VAE(16,config.latent_dim,16).to(device)

    optimizer = torch.optim.AdamW(vae.parameters(), lr = 1e-4)
    recon_loss_list = []
    kl_loss_list = []
    avg_loss_list = []
    for epoch in range(config.epoch):
        # 训练阶段
        vae.train()
        train_loss = 0
        for batch_idx, (x,) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_recon, mu, log_var = vae(x)
            loss,recon_loss,kl_loss = vae_loss(x_recon, x, mu, log_var)
            print(f"Rec: {recon_loss.item():.3f} | KL: {kl_loss.item():.3f}")
            recon_loss_list.append(recon_loss.item())
            kl_loss_list.append(kl_loss.item())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_loss = train_loss / len(train_loader)
        avg_loss_list.append(avg_loss)
        print(f"Epoch [{epoch+1}/{config.epoch}], Train: {avg_loss:.4f}")
    torch.save(vae.state_dict(), "vae.pth")
   # plot_loss(epochs=config.epoch,batch_size = train_loader.batch_size, kl_loss=(kl_loss_list,'red','-'),recon_loss=(recon_loss_list,'blue','-'),avg_loss=(avg_loss_list,'green','-'))
