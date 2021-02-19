import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='G:\python project\deep learning\example\MNIST_data',
    train=True,
    transform=torchvision.transforms.ToTensor(),  # (0-255)变(0-1)
    download=DOWNLOAD_MNIST
)

# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title(f'{train_data.train_labels[0]}')
# plt.show()

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

data = torchvision.datasets.MNIST(
    root='G:\python project\deep learning\example\MNIST_data',
    train=False
)
test_x = torch.unsqueeze(data.data, dim=1).type(torch.FloatTensor)[:2000] / 255.    # 图
test_y = data.targets[:2000]    # 分类

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # (1, 28, 28)
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2  # (pic_size+2*padding-kernel_size)/stride+1=pic_size
            ),  # (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2)  # (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # (batch, 32, 7, 7)
        x = x.view(x.size(0), -1)  # (batch, 32 * 7 * 7)
        x = self.out(x)
        return x


cnn = CNN()
# print(cnn)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        output = cnn(batch_x)
        loss = loss_func(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
            accuracy = sum(pred_y == test_y.numpy())/test_y.size(0)
            print(f'Epoch: {epoch}| train loss: {loss.item():.3f}| test accuracy: {accuracy}')

test_output = cnn(test_x[10:20])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[10:20].numpy(), 'real number')


