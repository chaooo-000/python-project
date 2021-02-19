import torch

x = torch.FloatTensor([[1, 2], [3, 4]])
x.requires_grad_()  # 让requires_grad=True

print(x.size())
x_out = torch.mean(x * x)
print(x_out)
x_out.backward()
print(x.grad)
