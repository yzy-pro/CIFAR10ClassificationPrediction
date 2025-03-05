import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import CIFAR10  # 使用 CIFAR-10 数据集
from model import LeNet  # 假设模型在 `model.py` 文件中


def test_data_process():  # 训练——验证下载、处理函数
    # CIFAR-10 数据集加载
    test_data = CIFAR10(root='./data/test',  # 修改为 CIFAR-10 数据集
                        train=False,
                        transform=transforms.Compose(
                            [transforms.Resize(size=32),
                             transforms.ToTensor()]),  # 将图像大小设置为 32x32
                        download=True)

    # 数据打包
    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=2)
    return test_dataloader


def test_model_process(model, test_dataloader):
    # 检测设备
    device = "xpu" if torch.xpu.is_available() else 'cpu'
    # 模型放到设备中
    model = model.to(device)

    # 初始化参数
    test_corrects = 0.0
    test_num = 0

    # 只进行前向传播，不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            # 数据放到设备中
            test_data_x = test_data_x.to(device)
            # 标签放到设备中
            test_data_y = test_data_y.to(device)
            # 设置模型为评估模式
            model.eval()
            # 前向传播过程，输入为测试数据集，输出为对每个样本的预测值
            output = model(test_data_x)
            # 查找每一行最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 若预测正确，则准确度 test_corrects 加 1
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            # 将所有的测试样本累加
            test_num += test_data_x.size(0)
        # 计算准确率
        test_acc = test_corrects.double().item() / test_num
        print("准确率：", test_acc)


if __name__ == "__main__":
    # 加载模型
    model = LeNet()

    # 加载已训练的最佳模型权重
    model.load_state_dict(torch.load('module/best_model.pth'))

    # 加载测试数据
    test_dataloader = test_data_process()

    # 加载模型测试的函数
    test_model_process(model, test_dataloader)

    # 检测设备并将模型转移到设备
    device = "xpu" if torch.xpu.is_available() else 'cpu'
    model = model.to(device)

    # CIFAR-10 类别
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
               'horse', 'ship', 'truck']

    # 模型预测部分
    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 设置模型为验证模式
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            result = pre_lab.item()
            label = b_y.item()
            print(
                f"预测值为: {classes[result]} ------- 真实值： {classes[label]}")
