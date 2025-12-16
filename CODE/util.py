import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt

class Dictionary_to_Object(dict):
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Dictionary_to_Object(value))
            else:
                setattr(self, key, value)

    def __getattr__(self,name):
        return None


def instantiate_from_config(path):
    with open(path) as f:
            config = yaml.safe_load(f)
    return Dictionary_to_Object(config['model'])


def normalize(batch):
    """
    输入：张量形状 (B, C, H, W)
    输出：归一化后的张量及每个通道的min/max值
    """
    # 每个通道独立计算min/max（沿batch、高度、宽度维度）
    min_vals = torch.amin(batch,  dim=(0, 2, 3), keepdim=True)  # shape (1, C, 1, 1)
    max_vals = torch.amax(batch,  dim=(0, 2, 3), keepdim=True)

    # Max-Min归一化到[-1,1]
    normalized = 2 * (batch - min_vals) / (max_vals - min_vals) - 1
    return normalized



def plot_loss(epochs,batch_size, **kwargs):
    """
    Parameters:
    epochs: 训练轮次
    kwargs: 接收两种类型的参数：
        - 普通损失曲线：参数名格式为"{label}_loss"，如 recon_loss=(data, 'red', '--')
        - 平均损失曲线:avg_loss=(data, 'blue', '-')
    """
    # 坐标轴生成逻辑
    x1 = np.arange((800//batch_size)*epochs)   # 每个epoch含25个batch的坐标
    x2 = np.arange(0,  (800//batch_size)*epochs, (800//batch_size))  # 每个epoch结束点坐标

    # 解构参数
    avg_data = None
    avg_config = {'label':'Avg Loss', 'color':'blue', 'linestyle':'-'}

    # 遍历所有参数
    for key, value in kwargs.items():
        if key == 'avg_loss':
            avg_data = value[0]
            # 允许自定义样式
            if len(value) > 1: avg_config['color'] = value[1]
            if len(value) > 2: avg_config['linestyle'] = value[2]
        else:
            # 解析标签名称（将参数名中的_替换为空格）
            label = key.replace('_',  ' ')
            config = {
                'label': label,
                'color': value[1] if len(value)>1 else None,
                'linestyle': value[2] if len(value)>2 else '-'
            }
            # 绘制普通损失曲线
            plt.plot(x1,  value[0],
                    label=config['label'],
                    color=config['color'],
                    linestyle=config['linestyle'])

    # 绘制平均损失曲线
    if avg_data is not None:
        plt.plot(x2,  avg_data,
                label=avg_config['label'],
                color=avg_config['color'],
                linestyle=avg_config['linestyle'])

    # 图表装饰
    plt.title(f'Loss  Curves ({epochs} Epochs)')
    plt.xlabel('Training  Steps')
    plt.ylabel('Loss  Value')
    plt.legend()
    plt.grid(True)
    #plt.savefig('ldm_train.png',  dpi=300)
    plt.show()


def plot_cdf(*datasets,x):
    plt.figure(figsize=(10, 6))
    max_error = 0
    colors = ['blue', 'green', 'red', 'purple', 'orange']  # 预定义颜色区分不同k值

    for idx, (dis, k) in enumerate(datasets):
        sorted_errors = np.sort(dis)
        cdf = np.arange(1, len(sorted_errors)+1) / len(sorted_errors)
        color = colors[idx % len(colors)]  # 循环使用颜色
        plt.plot(sorted_errors, cdf, linewidth=1, label=k, color=color)
        current_max = np.max(sorted_errors)
        max_error = max(max_error, current_max)

    plt.xlabel('定位误差 (米)', fontsize=12)
    plt.ylabel('误差累积概率', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, max_error)
    plt.yticks(np.arange(0,  1.1, 0.1))
    plt.legend(loc='lower right')  # 显示图例
    #plt.show()
    plt.savefig(f'cdf(k = {x}).png',  dpi=300)
