import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import torchvision
from torch.utils.data import DataLoader
import numpy as np

# CIFAR-10 标签名称
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 1. 加载数据集
test_data_set = torchvision.datasets.CIFAR10(
    root="../data/test",
    train=False,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010]),
    ]),
    download=True,
)

test_loader = DataLoader(test_data_set, batch_size=100, shuffle=False)

# 2. 加载训练好的模型
model_path = '../models/E_22_acc_0.8150040064102564.pth'
device = torch.device(
    'xpu' if torch.xpu.is_available() else 'cpu')  # 使用GPU或CPU

# 加载模型
model = torch.load(model_path, map_location=device)
model.eval()  # 设置为评估模式

# 3. 进行模型预测，收集每个样本的真实标签和预测的概率
y_true = []
y_scores = []

# 遍历测试数据集
for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)

    # 模型预测
    outputs = model(images)

    # 获取概率
    probabilities = torch.softmax(outputs, dim=1)

    # 获取每个样本的真实标签和预测概率
    y_true.extend(labels.cpu().numpy())
    y_scores.extend(probabilities.cpu().detach().numpy())

# 将 y_true 和 y_scores 转换为 numpy 数组
y_true = np.array(y_true)
y_scores = np.array(y_scores)

# 4. 计算 Precision-Recall 曲线和平均精度
average_precision = average_precision_score(y_true, y_scores, average="macro")

# 绘制 PR 曲线
plt.figure(figsize=(8, 6))

# 为每一个类绘制 PR 曲线
for i in range(10):
    precision, recall, _ = precision_recall_curve(y_true == i, y_scores[:, i])
    plt.plot(recall, precision,
             label=f'{class_names[i]} (AP={average_precision_score(y_true == i, y_scores[:, i]):.2f})')

# 设置图表标题和标签
plt.title('Precision-Recall Curve for CIFAR-10')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower left", bbox_to_anchor=(1.05, 0))
plt.grid(True)

# 保存图片到当前文件夹
plt.tight_layout()
plt.savefig('precision_recall_curve.png')

# 显示图形
plt.show()
