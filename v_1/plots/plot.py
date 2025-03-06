from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt

# 图像预处理，方便后续处理
train_data = CIFAR10(root='../../data/train',
                     train=True,
                     transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),  # 调整为224x224
                     download=True)

# 数据打包，以64为一组捆起来
train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=64,
                               shuffle=True,
                               num_workers=0)

# 获得一个Batch的数据标签，并将其打印
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
batch_x = b_x.permute(0, 2, 3, 1).numpy()  # 将图像张量从 (batch_size, channels, height, width) 转为 (batch_size, height, width, channels)
batch_y = b_y.numpy()  # 将标签转换为Numpy数组
class_label = train_data.classes  # CIFAR-10的类别标签

# 可视化一个Batch的图像
plt.figure(figsize=(12, 5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4, 16, ii + 1)
    plt.imshow(batch_x[ii], cmap=plt.cm.bone)  # 使用默认的cmap，不用灰度显示
    plt.title(class_label[batch_y[ii]], size=10)  # 使用CIFAR-10的类别标签
    plt.axis("off")
    plt.subplots_adjust(wspace=0.05)
plt.savefig("plot.png")
plt.show()
