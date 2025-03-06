import torch
import torchvision
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# CIFAR-10 标签名称
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# 1. 加载数据集
test_data_set = torchvision.datasets.CIFAR10(
    root="../data/test",
    train=False,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            ),
        ]
    ),
    download=True,
)

test_loader = DataLoader(test_data_set, batch_size=100, shuffle=False)

# 2. 加载训练好的模型
model_path = "../models/E_22_acc_0.8150040064102564.pth"
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")  #
# 判断是否有GPU，如果有则使用GPU，否则使用CPU
model = torch.load(model_path, map_location=device)  # 加载模型并确保在正确的设备上
model.eval()  # 将模型设置为评估模式

# 3. 获取所有测试集的预测结果
all_labels = []
all_preds = []

# 不进行梯度计算，节省内存
with torch.no_grad():
    for images, labels in test_loader:
        # 将图像和标签移到对应的设备
        images, labels = images.to(device), labels.to(device)

        # 预测
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        # 保存预测和真实标签
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# 4. 生成分类报告（Precision, Recall, F1 Score）
report = classification_report(
    all_labels, all_preds, target_names=class_names, output_dict=True
)

# 转化为 DataFrame，便于绘图
report_df = pd.DataFrame(report).transpose()

# 5. 打印 report_df，查看其结构
print(report_df)
report_df = report_df.drop(columns=["support"])
# 6. 删除 'accuracy' 列之前，先检查它是否存在
if "accuracy" in report_df.columns:
    report_df.drop("accuracy", axis=1, inplace=True)

# 7. 绘制 Precision, Recall 和 F1 Score 图
report_df.plot(kind="bar", figsize=(10, 7))
plt.title("Classification Report (Precision, Recall, F1 Score)")
plt.ylabel("Score")
plt.xlabel("Classes")
plt.xticks(rotation=45)
plt.tight_layout()

# 保存图像
plt.savefig("classification_report.png")  # 保存到当前文件夹
plt.show()
