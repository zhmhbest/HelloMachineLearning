"""
    Pytorch的Sequential不像Keras那样可以自动计算上层的输出维度作为下层的输入
"""
from torch.nn import Module, Sequential
from torch.nn import Conv2d, ReLU
from collections import OrderedDict

model1 = Sequential(
    Conv2d(1, 20, 5),
    ReLU(),
    Conv2d(20, 64, 5),
    ReLU()
)
print(model1)

model2 = Sequential(OrderedDict([
    ('conv1', Conv2d(1, 20, 5)),
    ('relu1', ReLU()),
    ('conv2', Conv2d(20, 64, 5)),
    ('relu2', ReLU())
]))
print(model2)

model3 = Sequential()
model3.add_module('conv1', Conv2d(1, 20, 5))
model3.add_module('relu1', ReLU())
model3.add_module('conv2', Conv2d(20, 64, 5))
model3.add_module('relu2', ReLU())
print(model3)


class MyModel(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 20, 5)
        self.relu1 = ReLU()
        self.conv2 = Conv2d(20, 64, 5)
        self.relu2 = ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x


model4 = MyModel()
print(model4)

model = model4
# 模型的所有可训练参数
for it in model.parameters():
    # 每层可训练参数
    print(it.size())
