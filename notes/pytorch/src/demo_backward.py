import torch
from torch.nn import MSELoss
from torch.nn import Sequential
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


def demo_backward():
    # 生成1张单通道32×32像素的图片
    data_x = torch.randn(1, 1, 32, 32)
    # data_y = torch.randn(10).view(1, -1)

    # 前向传播
    pred_y = model(data_x)

    # 反向传播
    pred_y.backward(torch.randn(1, 10))
    print("RAND", "channel[0].weight.grad[0][0][0]:", model[0].weight.grad[0][0][0])

    # 梯度置为0
    model.zero_grad()
    print("ZERO", "channel[0].weight.grad[0][0][0]:", model[0].weight.grad[0][0][0])


def demo_loss_backward():
    data_x = torch.randn(1, 1, 32, 32)
    data_y = torch.randn(10).view(1, -1)
    pred_y = model(data_x)

    # 损失函数
    loss_fn = MSELoss()
    loss_val = loss_fn(data_y, pred_y)
    print("LOSS", loss_val)
    loss_val.backward()
    print("LOSS", "channel[0].weight.grad[0][0][0]:", model[0].weight.grad[0][0][0])


if __name__ == '__main__':
    demo_backward()
    demo_loss_backward()
