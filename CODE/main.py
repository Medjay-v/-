import torch
import numpy as np
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from util import instantiate_from_config
from dataset.dataset import LatentDataset
from ldm.vae import vae_train
from ldm.diffusion import FP_LDM
from util import normalize,plot_cdf
from ldm.vqvae import vqgan_train
from pylab import mpl
from compared.ESRGAN import esrgan_train,Generator
from compared.GPR import gpr_train

config = instantiate_from_config('./configs/config.yaml')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mpl.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams['axes.unicode_minus'] = False

def KNN(RSS_train,coords_train,RSS_test,coords_test,k):
    scaler = StandardScaler()
    RSS_train_scaled = scaler.fit_transform(RSS_train)
    RSS_test_scaled = scaler.transform(RSS_test)

    knn_reg = KNeighborsRegressor(
        n_neighbors = k,
        weights = 'distance',
        algorithm = 'auto'
    )

    knn_reg.fit(RSS_train_scaled, coords_train)

    # 预测位置
    pred = knn_reg.predict(RSS_test_scaled)

    distances = np.sqrt(np.sum((coords_test - pred)**2, axis=1))
    average_error = np.mean(distances)
    std_error = np.std(distances)

    #print(f"平均定位误差: {average_error:.2f} 米")
    #print(f"误差标准差: {std_error:.2f} 米")
    return distances,average_error

def model(step,k):
    '--------------------初始化----------------------------'
    ldm = FP_LDM()
    dataset = LatentDataset(config.hr_path)
    hr = dataset.hr_test_tensor.unsqueeze(0)        #torch.Size([1, 16, 32, 32])
    lr = dataset.lr_test_tensor.unsqueeze(0)       #torch.Size([1, 16, 16, 16])
    lr_norm = normalize(lr)
    '--------------------训练部分----------------------------'
    if step == 1 :
        #vqgan_train()
        #vae_train()
        generated_hr = ldm.ldm_generate(lr_norm)
        # with torch.no_grad():
        #     generator = Generator()
        #     generator.load_state_dict(torch.load("ESRGAN.pth",weights_only=False))
        #     esrgan_hr = generator(lr_norm)
    elif step == 2 :
        #ldm.ldm_train()
        trained_generator = esrgan_train()
        #torch.save(trained_generator.state_dict(), "final_generator.pth")
    elif step == 3 :
        '----------------------------模型推理部分----------------------------'
        generated_hr = ldm.ldm_generate(lr_norm)     # torch.Size([1, 16, 32, 32]), 归一化数据
        '----------------------------双线性插值----------------------------'
        bilinear_hr = F.interpolate(            # torch.Size([1, 16, 32, 32]), 归一化数据
            lr_norm,
            size=(32, 32),
            mode='bilinear',
            align_corners=True
        )
        '----------------------------ESRGAN----------------------------'
        # with torch.no_grad():
        #     generator = Generator()
        #     generator.load_state_dict(torch.load("ESRGAN.pth",weights_only=False))
        #     esrgan_hr = generator(lr_norm)

        '----------------------------GPR----------------------------'
        # gpr_hr = gpr_train()
        # gpr_hr[:,::2,::2] = lr
        '--------------------本文算法生成图展示----------------------------'
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 3, 1)
        # plt.imshow(generated_hr.cpu().numpy().squeeze(0).transpose(1, 2, 0)[:, :, :1])
        # plt.title("Generated HR")
        # plt.subplot(1, 3, 2)
        # plt.imshow(bilinear_hr.cpu().numpy().squeeze(0).transpose(1, 2, 0)[:, :, :1])
        # plt.title("bilinear HR")
        # plt.subplot(1, 3, 3)
        # plt.imshow(hr.cpu().numpy().squeeze(0).transpose(1, 2, 0)[:, :, :1])
        # plt.title("real HR")
        # plt.savefig('result.png')
        # plt.show()

        '----------------------------数据反归一化----------------------------'
        min_vals = torch.amin(hr,  dim=(0, 2, 3), keepdim=True)
        max_vals = torch.amax(hr,  dim=(0, 2, 3), keepdim=True)
        #esrgan_hr[:,:,::2,::2] = lr
        generated_hr_denorm = (generated_hr.cpu() + 1) / 2 * (max_vals - min_vals) + min_vals     #torch.Size([1, 16, 32, 32])
        generated_hr_denorm[:,:,::2,::2] = lr
        bilinear_hr_denorm = (bilinear_hr  + 1) / 2 * (max_vals - min_vals) + min_vals       #torch.Size([1, 16, 32, 32])
        bilinear_hr_denorm[:,:,::2,::2] = lr
        '--------------------生成误差热力图----------------------------'
        # error_map = torch.abs(hr  - esrgan_hr).mean(dim=1)
        # error_map = error_map.squeeze()
        # error_np = error_map.cpu().detach().numpy()
        # fig = plt.figure(figsize=(10,8))
        # heatmap = plt.imshow(error_np,  cmap='gray', interpolation='nearest')
        # plt.xticks(range(0,32,4))   # 每4个像素显示一个刻度
        # plt.yticks(range(0,32,4))
        # plt.xlabel("X 坐标", fontsize=12)
        # plt.ylabel("Y 坐标", fontsize=12)
        # cbar = plt.colorbar(heatmap)
        # cbar.set_label(' 误差值', rotation=270, labelpad=15)
        # plt.title(" 像素级平均通道误差热力图", fontsize=14, pad=20)
        # plt.show()

        '----------------------------KNN定位部分----------------------------'
        '-------------索引------------'
        selected = [i * 32 + j + 1 for i in range(0, 32, 3) for j in range(0, 32, 3)]
        dots = set(range(1024))
        test_dots = set(selected)
        train_dots = sorted(dots - test_dots)
        test_dots = sorted(test_dots)

        '-------------坐标-------------'
        x,y = np.meshgrid(np.linspace(0,7.75,32), np.linspace(0,7.75,32),indexing='ij')
        coords = np.stack([x.T,  y.T], axis=-1)
        coords_lr = coords[::2,::2,:]
        coords = coords.reshape(-1, 2)
        coords_lr = coords_lr.reshape(-1,2)
        coords_train = coords[train_dots]
        coords_test = coords[test_dots]
        '-------------KNN训练集-------------'
        generated_hr = generated_hr_denorm.numpy().squeeze(0).transpose(1, 2, 0).reshape(-1, 16)
        hr = hr.numpy().squeeze(0).transpose(1, 2, 0).reshape(-1, 16)
        lr = lr.numpy().squeeze(0).transpose(1, 2, 0).reshape(-1, 16)
        bilinear_hr = bilinear_hr_denorm.numpy().squeeze(0).transpose(1, 2, 0).reshape(-1, 16)
        #esrgan_hr = esrgan_hr.numpy().squeeze(0).transpose(1, 2, 0).reshape(-1, 16)
        #gpr_hr = gpr_hr.cpu().numpy().transpose(1, 2, 0).reshape(-1, 16)
        '-------------KNN测试集-------------'
        RSS_test = hr[test_dots]

        '-------------------------------------------------------定位测试部分-------------------------------------------------------'
        """
            原始指纹库,稀疏指纹库,diffusion指纹库
        """
        # origin_dis,origin_avg = KNN(hr[train_dots],coords_train,RSS_test,coords_test,k)
        # lr_dis,lr_avg = KNN(lr,coords_lr,RSS_test,coords_test,k)
        # diffusion_dis,diffusion_avg = KNN(generated_hr[train_dots],coords_train,RSS_test,coords_test,k)
        # bilinear_dis,bilinear_avg = KNN(bilinear_hr,coords,RSS_test,coords_test,k)
        #esrgan_dis,esr_avg = KNN(esrgan_hr,coords,RSS_test,coords_test,k)
        #gpr_dis,gpr_avg = KNN(gpr_hr,coords,RSS_test,coords_test,k)

        #plot_cdf((origin_dis,'原始指纹库'),(lr_dis,'稀疏指纹库'),(diffusion_dis,'本文算法'),(bilinear_dis,'双线性插值'),(esrgan_dis,'ESRGAN'),(gpr_dis,'GPR'),x = k)
        # print(f"原始指纹库平均定位误差: {origin_avg:.2f} 米")
        # print(f"稀疏指纹库平均定位误差: {lr_avg:.2f} 米")
        # print(f"本文算法平均定位误差: {diffusion_avg:.2f} 米")
        # print(f"双线性插值平均定位误差: {bilinear_avg:.2f} 米")
        #print(f"ESRGAN平均定位误差: {esr_avg:.2f} 米")
        #print(f"GPR平均定位误差: {gpr_avg:.2f} 米")


if __name__ == '__main__':
    #for i in range(1,11):
        model(1,3)
