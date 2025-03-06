import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

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
model = torch.load(model_path, map_location=torch.device("cpu"))
model.eval()  # 切换模型为评估模式

# 3. 获取真实标签和模型预测标签
all_labels = []
all_preds = []

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, preds = torch.max(outputs, 1)  # 获取最大值索引作为预测标签
        all_labels.extend(labels.numpy())
        all_preds.extend(preds.numpy())

# 4. 计算混淆矩阵
cm = confusion_matrix(all_labels, all_preds)

# 5. 可视化混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=test_data_set.classes,
    yticklabels=test_data_set.classes,
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.savefig("./confusion_matrix.png")
plt.show()
