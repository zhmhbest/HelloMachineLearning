import torch
from torch import nn
from torch.nn import Module, Sequential
from torch.nn import ReLU

# in_channels,
# out_channels,
# kernel_size: int | (int, int),
# stride: int | (int, int) = 1,
# padding: int | (int, int) = 0
from torch.nn import Conv2d

# kernel_size: int | (int, int),
# stride: int | (int, int) = 1
# padding: int | (int, int) = 0
from torch.nn import MaxPool2d

# in_features: int,
# out_features: int,
# bias: bool = True
from torch.nn import Linear

# start_dim: int = 1,
# end_dim: int = -1
from torch.nn import Flatten

model = Sequential(
    Conv2d(1, 6, 5),
    ReLU(),
    MaxPool2d(2),
    Conv2d(6, 16, 5),
    ReLU(),
    MaxPool2d(2),
    Flatten(),
    Linear(16 * 5 * 5, 120),
    ReLU(),
    Linear(120, 84),
    ReLU(),
    Linear(84, 10)
)

# 生成1张单通道32×32像素的图片
data = torch.randn(1, 1, 32, 32)

# 前向传播
out = model(data)

# 反向传播（参数梯度缓存器置零，使用随机梯度）
model.zero_grad()
out.backward(torch.randn(1, 10))
