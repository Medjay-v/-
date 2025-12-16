import torch
import torch.nn as nn
import torch.nn.functional as F
from util import plot_loss,instantiate_from_config
from torch.utils.data import DataLoader
from dataset.dataset import LatentDataset
from ldm.vae import VAE
from ldm.vqvae import VQGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = instantiate_from_config('./configs/config.yaml')

"""
    类名：      噪声调度器
    初始化：    扩散总时间步、噪声调度器类型
    前向输入：  加噪前的图像、随机时间步
    前向输出：  加噪后的图像、噪声
    功能：      得到扩散模型的一系列α值
"""
class BetaScheduler:
    def __init__(self, num_timesteps,beta_schedule):
        self.timesteps = num_timesteps

        if beta_schedule == "linear":
            self.betas = torch.linspace(0.0001, 0.02, num_timesteps)
        elif beta_schedule == "cosine":
            s = 0.008
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
            self.betas = self.betas.to(device)

        self.alphas = 1. - self.betas
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def add_noise(self, x0, t):
        noise = torch.randn_like(x0)
        return self.sqrt_alphas_cumprod[t][:,None,None,None] * x0 + self.sqrt_one_minus_alphas_cumprod[t][:,None,None,None] * noise,noise



"""
    类名：      时间步嵌入
    初始化：    时间步嵌入深度
    前向输入：  时间步
    前向输出：  时间步嵌入张量
    功能：      通过正余弦编码和多层感知机处理,将时间步t转换为具有dim维度的时间步嵌入向量以便在扩散模型中使用
"""
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        half = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.proj(emb)



"""
    类名：      条件编码器
    初始化：    无
    前向输入：  低分辨率图像(16,32,32)
    前向输出：  潜在变量(latent_dim,8,8)
    功能：      将低分辨率图像尺寸转变为与潜在变量尺寸大小相同使其能够条件注入
"""
class ConditionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 64, 3, stride=2, padding=1),       # (32,8,8)
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),      # (64,4,4)
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),     # (128,2,2)
            nn.GroupNorm(8, 256),
            nn.SiLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2, padding=1),    # (128,1,1)
            nn.GroupNorm(8, 256),
            nn.SiLU(),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        return [x1,x2,x3,x4]



class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(query_dim, num_heads)
        self.norm = nn.LayerNorm(query_dim)  # 确保归一化维度匹配通道数
        self.proj = nn.Conv2d(query_dim, query_dim, kernel_size=1)  # 使用卷积保持空间维度
        self.context_proj = nn.Conv2d(context_dim, query_dim, 1)

    def forward(self, x, context):
        batch, c, h, w = x.shape
        context = self.context_proj(context)
        # 将输入展平为 [seq_len, batch, c]
        x_flat = x.view(batch, c, -1).permute(2, 0, 1)  # [h*w, batch, c]
        context_flat = context.view(batch, c, -1).permute(2, 0, 1)  # [cond_seq_len, batch, c]

        # 注意力计算
        attn_out, _ = self.attn(x_flat, context_flat, context_flat)
        attn_out = attn_out.permute(1, 2, 0).view(batch, c, h, w)  # 恢复形状 [batch, c, h, w]

        # 调整维度顺序以适配LayerNorm: [batch, c, h, w] -> [batch, h, w, c]
        attn_out = attn_out.permute(0, 2, 3, 1)
        attn_out = self.norm(attn_out)  # LayerNorm作用于最后一个维度c
        attn_out = attn_out.permute(0, 3, 1, 2)  # 恢复为 [batch, c, h, w]

        return self.proj(attn_out) + x



"""
    类名：      残差块
    初始化：    输入通道、输出通道、时间步嵌入深度
    前向输入：
    前向输出：
    功能：
"""
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.cross_attn = CrossAttention(out_channels, cond_dim)

        # 主卷积路径
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1))

        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1))

        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb, cond):
        h = self.conv1(x)
        # 时间嵌入融合
        t_emb = self.time_mlp(t_emb)
        h += t_emb[:, :, None, None]
        h = self.cross_attn(h, cond)
        h = self.conv2(h)
        return h + self.res_conv(x)



class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_dim):
        super().__init__()
        self.res_block = ResidualBlock(in_channels, out_channels, time_emb_dim,cond_dim)
        self.down = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x, t_emb,cond):
        x = self.res_block(x, t_emb, cond)
        return self.down(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1)
        self.res_block = ResidualBlock(in_channels, out_channels, time_emb_dim, cond_dim)

    def forward(self, x, skip, t_emb, cond):
        x = self.up(x)
        # print(x.shape)
        # print(skip.shape)
        x = torch.cat([x, skip], dim=1)

        return self.res_block(x, t_emb, cond)


"""
    类名：      Unet模型
    初始化：    隐变量维度、时间步嵌入维度
    前向输入：  特征图、时间步、条件变量特征图
    前向输出：  特征图
    功能：      用于去噪任务和图像生成
"""
class UNet(nn.Module):
    def __init__(self, latent_dim, time_emb_dim,unet_dim = 64):
        super().__init__()
        self.input = nn.Conv2d(latent_dim, unet_dim, 3, padding=1)

        self.time_emb = TimeEmbedding(time_emb_dim)

        self.cond_encoder = ConditionEncoder()

        cond_dim = [64, 128, 256, 256]

        # 下采样
        self.down1 = DownBlock(unet_dim, 128, time_emb_dim, cond_dim[0])
        self.down2 = DownBlock(128, 256, time_emb_dim, cond_dim[1])
        self.down3 = DownBlock(256, 512, time_emb_dim, cond_dim[2])

        # 中间层
        self.mid = ResidualBlock(512, 512, time_emb_dim, cond_dim[3])

        # 上采样
        self.up1 = UpBlock(512, 256, time_emb_dim, cond_dim[2])
        self.up2 = UpBlock(256, 128, time_emb_dim, cond_dim[1])
        self.up3 = UpBlock(128, unet_dim, time_emb_dim, cond_dim[0])

        # 最终输出
        self.final_conv = nn.Conv2d(unet_dim, latent_dim, 3, padding=1)

    def forward(self, x, t, cond):                # x = (latent_dim,8,8)
        t_emb = self.time_emb(t)
        cond = self.cond_encoder(cond)
        # 下采样
        x = self.input(x)                         # x : (latent_dim,8,8) -> (64,8,8)
        d1 = self.down1(x, t_emb, cond[0])        # cond[0] = (64,8,8)  =>   d1 = (128,4,4)
        d2 = self.down2(d1, t_emb, cond[1])       # cond[1] = (128,4,4) =>   d2 = (256,2,2)
        d3 = self.down3(d2, t_emb, cond[2])       # cond[2] = (256,2,2) =>   d3 = (512,1,1)
        # 中间处理
        mid = self.mid(d3, t_emb, cond[3])        # cond[3] = (512,1,1) => mid = (512,1,1)
        # 上采样

        u1 = self.up1(mid, d2, t_emb, cond[2])    # 上采样 mid = (256,2,2)   concat = (256,2,2) -> (512,2,2) -> u1 = (256,2,2)
        u2 = self.up2(u1, d1, t_emb, cond[1])     # 上采样 u1 = (128,4,4)    concat = (128,4,4) -> (256,4,4) -> u2 = (128,4,4)
        u3 = self.up3(u2, x, t_emb, cond[0])      # 上采样 u2 = (64,8,8)     concat = (64,8,8) -> (128,8,8)

        return self.final_conv(u3)



class FP_LDM():
    def __init__(self):
        # 数据集
        self.dataset = LatentDataset(config.hr_path)
        train_tensor = self.dataset.train_tensorDataset
        self.train_loader = DataLoader(train_tensor, batch_size=64, shuffle=True, pin_memory=True, drop_last=True)

        # 加载预训练VAE
        self.vae = VAE(16, config.latent_dim, 16).to(device)
        self.vae.eval()
        self.vae.load_state_dict(torch.load("vae.pth",weights_only=False))
        # self.vqgan = VQGAN(16, config.latent_dim,config.num_embedding).to(device)
        # self.vqgan.eval()
        # self.vqgan.load_state_dict(torch.load("vqgan.pth",weights_only=False))
        self.unet = UNet(config.latent_dim, config.time_dim).to(device)
        self.betaScheduler = BetaScheduler(1000, config.beta_schedule)
        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr = 1e-4, weight_decay=1e-5)

    def ldm_train(self):
        loss_list = []
        avg_loss_list = []
        self.unet.train()
        for epoch in range(config.epoch):
            total_loss = 0.0
            for batch_idx, (hr, lr) in enumerate(self.train_loader):
                hr = hr.to(device)
                lr = lr.to(device)
                self.optimizer.zero_grad()
                # 通过VAE编码获得隐变量
                with torch.no_grad():
                    h = self.vae.encoder(hr)
                    mu = self.vae.fc_mu(h)
                    log_var = self.vae.fc_logvar(h)
                    z = self.vae.reparameterize(mu, log_var)
                    #z = self.vqgan.encoder(hr)
                # 扩散加噪
                t = torch.randint(0, len(self.betaScheduler.betas), (hr.size(0),), device=device)
                z_t,noise = self.betaScheduler.add_noise(z,t)
                # 预测噪声
                pred_noise = self.unet(z_t, t, lr)
                loss = F.mse_loss(pred_noise, noise)
                print(f"Train Loss: {loss:.4f}")
                loss_list.append(loss.item())
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            avg_loss = total_loss / len(self.train_loader)
            avg_loss_list.append(avg_loss)
            print(f"Epoch [{epoch+1}/{config.epoch}], Train: {avg_loss:.4f}")
        torch.save(self.unet.state_dict(), "unet.pth")
     #   plot_loss(epochs=config.epoch, batch_size = self.train_loader.batch_size, loss=(loss_list,'red','-'),avg_loss=(avg_loss_list,'green','-'))


    @torch.no_grad()
    def ldm_generate(self,lr):
        self.unet.load_state_dict(torch.load("unet.pth",weights_only=False))
        self.unet.eval()
        lr = lr.to(device)
        z = torch.randn((1, config.latent_dim, 8, 8), device=device)

        for t in reversed(range(len(self.betaScheduler.betas))):
            timesteps = torch.full((1,), t, device=device, dtype=torch.long)
            pred_noise = self.unet(z, timesteps, lr)

            alpha_t = self.betaScheduler.alphas[t]
            alpha_cumprod_t = self.betaScheduler.alphas_cumprod[t]
            beta_t = self.betaScheduler.betas[t]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)

            #μ = 1 / sqrt_alpha_t * (z - beta_t * pred_noise / sqrt_one_minus_alpha_cumprod_t)
            μ = (z - sqrt_one_minus_alpha_cumprod_t * pred_noise) / sqrt_alpha_t
            noise = torch.randn_like(z) if t > 0 else 0
            z =  sqrt_alpha_t * μ + torch.sqrt(beta_t) * noise

        # 通过VAE解码生成高分辨率图像
        with torch.no_grad():
            #generated_hr = self.vqgan.decoder(z)
            h = self.vae.decoder_fc(z)
            generated_hr = self.vae.decoder(h)
        return generated_hr
