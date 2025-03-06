import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.preprocessing import label_binarize

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
model = torch.load(model_path, map_location=torch.device("cpu"))
model.eval()  # 切换模型为评估模式

# 3. 获取真实标签和模型预测的概率
all_labels = []
all_preds_prob = []

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)  # 模型输出的原始logits
        softmax_outputs = torch.softmax(outputs, dim=1)  # 转换为概率
        all_labels.extend(labels.numpy())
        all_preds_prob.extend(softmax_outputs.numpy())

# 4. 将标签转化为二进制格式，每个类别都有一个独立的标签
all_labels_bin = label_binarize(all_labels, classes=np.arange(10))

# 5. 计算每个类别的ROC曲线和AUC
fpr = {}
tpr = {}
roc_auc = {}

for i in range(10):  # CIFAR-10有10个类别
    fpr[i], tpr[i], _ = roc_curve(
        all_labels_bin[:, i], [prob[i] for prob in all_preds_prob]
    )
    roc_auc[i] = auc(fpr[i], tpr[i])

# 6. 绘制所有类别的ROC曲线
plt.figure(figsize=(10, 8))

for i in range(10):
    plt.plot(fpr[i], tpr[i], lw=2, label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")

# 绘制对角线（随机猜测的基准）
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")

plt.title("Receiver Operating Characteristic (ROC) Curve for CIFAR-10")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")

# 7. 保存ROC曲线图像
plt.savefig("./roc_curve.png")

# 如果需要显示图像，可以添加：
plt.show()
