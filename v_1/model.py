import torch
from torch import nn
from torchsummary import summary


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 修改卷积层1: CIFAR-10是3通道输入
        self.c1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, padding=2)
        # 激活函数
        self.sig = nn.ReLU()
        # 池化层1
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 修改卷积层2: 将输入通道调整为8（来自上一层的输出通道）
        self.c3 = nn.Conv2d(in_channels=8, out_channels=20, kernel_size=5)
        # 池化层2
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 平展层
        self.flatten = nn.Flatten()
        # 由于输入图像尺寸变为32x32，计算平展后输入全连接层的尺寸
        # (32-5+2*2)//2 = 16 -> (16-5+2*0)//2 = 6
        # 所以输出尺寸为 6x6x20
        self.f5 = nn.Linear(6 * 6 * 20, 120)  # 修改全连接层输入尺寸
        self.f6 = nn.Linear(120, 84)
        self.f7 = nn.Linear(84, 10)  # CIFAR-10有10个类别

        self.dropout = nn.Dropout(0.3)  # 随机丢弃神经元

    def forward(self, x):
        x = self.sig(self.c1(x))
        x = self.s2(x)
        x = self.sig(self.c3(x))
        x = self.s4(x)
        x = self.flatten(x)
        x = self.dropout(self.f5(x))  # 在全连接层后应用dropout
        x = self.f6(x)
        x = self.f7(x)
        return x


if __name__ == "__main__":
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    model = LeNet().to(device)
    # 将模型LeNet()放到设备中实例化成model
    print(summary(model, (3, 32, 32)))  # CIFAR-10的图片尺寸是32x32，3个通道
