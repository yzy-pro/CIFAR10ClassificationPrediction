import copy
import time
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from model import LeNet
from torchvision import datasets, transforms
import torch.utils.data as Data


def train_val_data_process():  # 训练——验证下载、处理函数
    # 加载CIFAR10数据集，resize到32x32，转换成Tensor
    transform = transforms.Compose([
        transforms.Resize(size=32),  # CIFAR-10的图片尺寸是32x32，调整为一致的大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # 数据标准化
    ])

    # 下载并加载CIFAR10数据集（训练集）
    train_data = datasets.CIFAR10(root='./data/train', train=True,
                                  transform=transform, download=True)

    # 划分训练集与验证集
    train_data, val_data = Data.random_split(train_data,
                                             [round(0.8 * len(train_data)),
                                              round(0.2 * len(
                                                  train_data))])  # 划分80%训练集，20%验证集

    # 数据加载器
    train_loader = Data.DataLoader(dataset=train_data,
                                   batch_size=32,
                                   shuffle=True,
                                   num_workers=2)
    val_loader = Data.DataLoader(dataset=val_data,
                                 batch_size=32,
                                 shuffle=True,
                                 num_workers=2)

    return train_loader, val_loader

def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")  #
    # 检查设备
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)  # 使用 Adam
    # 优化器，学习率为 0.01
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    model = model.to(device)  # 将模型移动到设备上
    best_model_wts = copy.deepcopy(model.state_dict())  # 复制当前模型参数

    best_acc = 0.0
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []
    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # 初始化参数
        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0
        train_num = 0
        val_num = 0

        # 训练阶段
        model.train()
        for step, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)

            loss = criterion(output, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(pre_lab == b_y.data)
            train_num += b_x.size(0)

        # 验证阶段
        model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 在验证时不计算梯度
            for step, (b_x, b_y) in enumerate(val_dataloader):
                b_x = b_x.to(device)
                b_y = b_y.to(device)

                output = model(b_x)
                pre_lab = torch.argmax(output, dim=1)

                loss = criterion(output, b_y)

                val_loss += loss.item() * b_x.size(0)
                val_corrects += torch.sum(pre_lab == b_y.data)
                val_num += b_x.size(0)

        # 计算并保存每一轮的损失数据和准确率
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        if val_num > 0:  # 确保 val_num 不为零
            val_loss_all.append(val_loss / val_num)
            val_acc_all.append(val_corrects.double().item() / val_num)
        else:
            val_loss_all.append(float('inf'))  # 或者选择其他合适的方式处理
            val_acc_all.append(0)

        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # 寻找最高准确度的参数
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        time_use = time.time() - since
        print("训练耗费时间：{:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))

    # 加载最高准确率下的模型参数
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, '../module/best_model.pth')

    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                        "train_loss_all": train_loss_all,
                                        "val_loss_all": val_loss_all,
                                        "train_acc_all": train_acc_all,
                                        "val_acc_all": val_acc_all})
    return train_process
def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, 'ro-', label= "train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, 'bs-', label= "val loss")
    plt.legend()
    plt.xticks(train_process["epoch"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, 'ro-',
             label="train acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'bs-',
             label="val acc")
    plt.xticks(train_process["epoch"])
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # 将模型实例化
    LeNet = LeNet()
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(LeNet, train_dataloader,
                                        val_dataloader, 50)
    matplot_acc_loss(train_process)


