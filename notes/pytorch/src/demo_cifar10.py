import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torch import optim
from torch.nn import Module
from torch.nn import ReLU, Flatten
from torch.nn import Conv2d, MaxPool2d, Linear
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import Dataset, DataLoader
from torch.utils.data.dataset import T_co
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

"""
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    加载数据
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""


def load_data():
    dataset_train = CIFAR10(root='./data', train=True, download=True)
    dataset_test = CIFAR10(root='./data', train=False, download=True)

    return (
        dataset_train.classes,
        dataset_train.data,
        dataset_train.targets,
        dataset_test.data,
        dataset_test.targets
    )


classes, x_train, y_train, x_test, y_test = load_data()
size_train = len(x_train)
size_test = len(x_test)

# 图片（高、宽、通道）
print(f"(H, W, C) = {x_train[0].shape}")
print(f"Size(train) = {size_train}")
print(f"Size(test) = {size_test}")


def show_images(num):
    global x_train, y_train
    print([classes[_i_] for _i_ in y_train[0:10]])
    plt.imshow(np.hstack(x_train[0:10]))
    plt.show()


# 显示头10张图片
# show_images(10)


"""
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    数据预处理
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""


class DataHolder(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, x_transform=None, y_transform=None) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.x_transform = x_transform
        self.y_transform = y_transform

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index) -> T_co:
        return (
            self.x[index] if self.x_transform is None else self.x_transform(self.x[index]),
            self.y[index] if self.y_transform is None else self.y_transform(self.y[index])
        )


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
loader_train = DataLoader(DataHolder(x_train, y_train, transform), batch_size=32, shuffle=True, num_workers=0)
loader_test = DataLoader(DataHolder(x_test, y_test, transform), batch_size=32, shuffle=False, num_workers=0)
# for i, (x_batch, y_batch) in enumerate(loader_train, 0):


"""
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    创建模型
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""


class MyModel(Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.relu1 = ReLU()
        self.conv1 = Conv2d(3, 6, 5)
        self.pool1 = MaxPool2d(2, 2)
        self.conv2 = Conv2d(6, 16, 5)
        self.flatten1 = Flatten()
        self.fc1 = Linear(16 * 5 * 5, 120)
        self.fc2 = Linear(120, 84)
        self.fc3 = Linear(84, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool1(self.relu1(self.conv2(x)))
        x = self.flatten1(x)
        x = self.relu1(self.fc1(x))
        x = self.relu1(self.fc2(x))
        x = self.fc3(x)
        return x


model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()


"""
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    训练模型
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""


def train(epoch, print_each=200):
    for ep in range(1, 1 + epoch):
        loss_running = 0.0
        for i, (x_batch, y_batch) in enumerate(loader_train, 0):
            # 前向传播
            optimizer.zero_grad()
            y_predict = model(x_batch)

            # 计算损失
            loss_val = criterion(y_predict, y_batch)

            # 反向传播及优化
            loss_val.backward()
            optimizer.step()

            loss_running += loss_val.item()
            if 1 == (print_each - (i % print_each)):
                print(f"epoch:{ep}, index:{i + 1}, loss:{loss_running / print_each}")
                loss_running = 0.0
    print('Finished Training')


train(10)


"""
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    预测
    ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
"""


def get_correction(loader: DataLoader):
    global size_train
    total = 0
    correction = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            y_predict = model(x_batch)
            _, predicted = torch.max(y_predict, 1)
            correct = predicted == y_batch
            assert isinstance(correct, torch.Tensor)
            correction += torch.sum(correct).item()
            total += y_batch.size(0)
        return correction / total


print(f"Accuracy of train: {get_correction(loader_train)}")
print(f"Accuracy of test : {get_correction(loader_test)}")

