# 何 2020/6/2 拟合， 6/5 保存提取
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # 'unsqueeze' convert data from 2D to 1D
y = x.pow(2) + 0.2 * torch.rand(x.size())

# class Net(torch.nn.Module):
#     def __init__(self, n_feature, n_hidden, n_output):
#         super(Net, self).__init__()
#         self.hidden = nn.Linear(n_feature, n_hidden)
#         self.out = nn.Linear(n_hidden, n_output)
# 
#     def forward(self, x):
#         x = F.relu(self.hidden(x))
#         x = self.out(x)
#         return x
# net = Net(1, 10, 1)

# 快速搭建
net = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

# 优化
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)   # parameters in the net, learning rate
loss_func = nn.MSELoss()

# 实时打印
plt.ion()
plt.show()

for t in range(100):
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 5 == 0:
        plt.cla()   # clear axis
        plt.scatter(x.numpy(), y.numpy())   # scatter points
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, f'loss={loss.item()}', fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

# 保存网络
torch.save(net, 'G:\python project\deep learning\example\\net and params\\regression_net.pkl')
# 保存网络参数
torch.save(net.state_dict(), 'G:\python project\deep learning\example\\net and params\\regression_net_params.pkl')

# 读取网络
# net2 = torch.load('G:\python project\deep learning\example\\net and params\\regression_net.pkl')
# 读取网络参数
# net2 = nn.Sequential(
#     nn.Linear(1, 10),
#     nn.ReLU(),
#     nn.Linear(10, 1)
# )
# net2.load_state_dict(torch.load('G:\python project\deep learning\example\\net and params\\regression_net_params.pkl'))

plt.ioff()
plt.show()
