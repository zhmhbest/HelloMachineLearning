import torch

x = torch.ones(2, 2, requires_grad=True)
y = 3 * torch.pow(x + 2, 2)
out = y.mean()
print(x)
print(y)
print(out)
out.backward(torch.tensor(1))
print(x.grad)
