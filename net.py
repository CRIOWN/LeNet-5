import torch
from torch import nn

# 定义一个网络模型

class MYLeNet5(nn.Module):
    # 初始化网络
    def __init__(self):
        super(MYLeNet5, self).__init__()
        # N=(W-F+2P)/S+1  输出大小+（输入-卷积核+2*填充值)/步长+1
        # 卷积层
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        # Sigmoid
        self.Sigmoid = nn.Sigmoid()
        #  池化层
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 卷积层
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 池化层
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 卷积层
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)

        self.flatten = nn.Flatten()

        # 全连接层
        self.f6 = nn.Linear(120, 84)
        # 全连接层
        self.output = nn.Linear(84, 10)


    def forward(self, x):
        x = self.Sigmoid(self.c1(x))
        x = self.s2(x)
        x = self.Sigmoid(self.c3(x))
        x = self.s4(x)
        x = self.c5(x)
        x = self.flatten(x)
        x = self.f6(x)
        x = self.output(x)
        return x


if __name__ == "__main__":
    x = torch.rand([1, 1, 28, 28])
    model = MYLeNet5()
    y = model(x)