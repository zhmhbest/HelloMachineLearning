import torch

# 创建张量
my_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
print(my_tensor)
print(torch.randn_like(my_tensor, dtype=torch.float))
print(torch.arange(0, 10))
print(torch.rand(5, 2, dtype=torch.float32))
print(torch.zeros(5, 2, dtype=torch.int32))
print(torch.ones(5, 2, dtype=torch.float32))
print(torch.empty(5, 2, dtype=torch.float64))  # 未初始化

# 张量的属性
print(my_tensor.size())
print(my_tensor.dim())
print(my_tensor.type())

# 简单计算
x = torch.tensor([1])
y = torch.tensor([2])
print(torch.add(x, y))
result = torch.empty(1, dtype=torch.float32)
torch.add(x, y, out=result)
print(result)

# 调整形状
arr = torch.arange(16)
print(arr)
print(arr.view(2, 8))
print(arr.view(-1, 8))
print(arr.view(2, -1))
print(arr.view(4, 4))
