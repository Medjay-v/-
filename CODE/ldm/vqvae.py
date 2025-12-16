import torch
import torch.nn as nn
import torch.nn.functional as F
from util import instantiate_from_config
from torch.utils.data import DataLoader
from dataset.dataset import LatentDataset
from torch.cuda.amp import autocast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = instantiate_from_config('./configs/config.yaml')

# 编码器实现
class Encoder(nn.Module):
    def __init__(self, in_channels=16, latent_dim=32):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            nn.Conv2d(128, latent_dim, 3, padding=1),
        )

    def forward(self, x):
        return self.conv_blocks(x)

# 解码器实现
class Decoder(nn.Module):
    def __init__(self, out_channels=16, latent_dim=32):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(latent_dim, 128, 3, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, out_channels, 4, stride=2, padding=1),  # 16x16 -> 32x32
        )

    def forward(self, x):
        return self.conv_blocks(x)

# 矢量量化层
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=32, beta=0.25):
        super().__init__()
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0/num_embeddings, 1.0/num_embeddings)
        self.beta = beta
        self.embedding_dim = embedding_dim

    def forward(self, z):
        # 展平空间维度
        z_flat = z.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        z_flattened = z_flat.view(-1, self.embedding_dim)  # [B*H*W, C]

        # 计算距离
        distances = torch.cdist(z_flattened, self.codebook.weight)

        # 获取最近邻编码
        encoding_indices = torch.argmin(distances, dim=1)
        quantized_flat = self.codebook(encoding_indices)

        # 重构量化后的张量
        quantized = quantized_flat.view(z_flat.shape).permute(0, 3, 1, 2).contiguous()

        # 计算量化损失
        commitment_loss = F.mse_loss(quantized.detach(), z)
        codebook_loss = F.mse_loss(quantized, z.detach())
        vq_loss = self.beta * commitment_loss + codebook_loss

        # 梯度直通估计
        quantized = z + (quantized - z).detach()
        return quantized, vq_loss, encoding_indices.view(z.shape[0], -1)

# 判别器实现
class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4),
            nn.Flatten()
        )

    def forward(self, x):
        return self.model(x)

class VQGAN(nn.Module):
    def __init__(self, in_channels=16, latent_dim=32, num_embeddings=512):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(in_channels, latent_dim)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim)
        self.discriminator = Discriminator(in_channels)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, _ = self.quantizer(z)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss



def configure_optimizers(model, learning_rate=2e-4):
    opt_vq = torch.optim.Adam(
        list(model.encoder.parameters()) +
        list(model.decoder.parameters()) +
        list(model.quantizer.parameters()),
        lr=learning_rate, betas=(0.5, 0.9)
    )
    opt_disc = torch.optim.Adam(
        model.discriminator.parameters(),
        lr=learning_rate/2, betas=(0.5, 0.9)
    )
    return opt_vq, opt_disc


def vqgan_train():
    dataset = LatentDataset(config.hr_path)
    hr_train_tensor = dataset.hr_tensorDataset
    train_loader = DataLoader(
        hr_train_tensor,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    vqgan = VQGAN(16,config.latent_dim,config.num_embedding).to(device)
    #optimizer = torch.optim.AdamW(vqgan.parameters(), lr = 1e-4)
    #optimizer_d = torch.optim.AdamW(vqgan.discriminator.parameters(), lr = 1e-4)
    opt_vq,opt_disc = configure_optimizers(vqgan)


    for epoch in range(config.epoch):
        # 训练阶段
        vqgan.train()
        for batch_idx, (x,) in enumerate(train_loader):
            real = x.to(device)
            opt_disc.zero_grad()
            with torch.no_grad():
                fake, _ = vqgan(real)
            disk_real = vqgan.discriminator(real)
            disc_fake  = vqgan.discriminator(fake.detach())

            loss_disc_real = F.binary_cross_entropy_with_logits(disk_real, torch.ones_like(disk_real))
            loss_disc_fake = F.binary_cross_entropy_with_logits(disc_fake , torch.zeros_like(disc_fake ))
            loss_disc = (loss_disc_real + loss_disc_fake) * 0.5
            loss_disc.backward()
            opt_disc.step()

            opt_vq.zero_grad()
            fake, vq_loss = vqgan(real)
            disc_fake = vqgan.discriminator(fake)
            loss_gan = F.binary_cross_entropy_with_logits(disc_fake, torch.ones_like(disc_fake))
            loss_recon = F.l1_loss(fake, real)

            # 总损失
            total_loss = loss_recon + vq_loss + 0.1 * loss_gan
            total_loss.backward()
            opt_vq.step()
    torch.save(vqgan.state_dict(), "vqgan.pth")



