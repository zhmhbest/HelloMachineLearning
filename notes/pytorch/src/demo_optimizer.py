import torch
from torch import optim
from torch.nn import MSELoss
from demo_backward import model  # 实例化的模型

data_x = torch.randn(1, 1, 32, 32)
data_y = torch.randn(10).view(1, -1)
loss_fn = MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer.zero_grad()

for i in range(3):
    # 前向传播
    pred_y = model(data_x)

    # 计算损失
    loss_val = loss_fn(data_y, pred_y)
    print(loss_val)

    # 反向传播
    loss_val.backward()

    # 优化权重
    optimizer.step()
