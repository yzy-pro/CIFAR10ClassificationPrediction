from torch import nn
import torch



class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,
                      stride=1, padding=1),# 卷积
            nn.ReLU(),# 取代sigmod
            nn.MaxPool2d(kernel_size=2, stride=2),# 最大池化
            nn.BatchNorm2d(num_features=32),# 标准化

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=128),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),# 铺平

            nn.Linear(in_features=128 * 4 * 4, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),# 神经元随机失活

            # nn.Linear(in_features=1024, out_features=512),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),

            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            # nn.Linear(in_features=256, out_features=128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),

            nn.Linear(in_features=256, out_features=10),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),

        )

    def forward(self, x):
        x = self.main(x)
        x= self.fc(x)
        return x

learning_rate = 1e-3
mynet = MyNet()
loss_fn = nn.CrossEntropyLoss() # 损失函数：交叉熵
optim = torch.optim.Adam(mynet.parameters(), lr=learning_rate)# Adam加速


if __name__ == '__main__':
    net = MyNet()
    print(net)