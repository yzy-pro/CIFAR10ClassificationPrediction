import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

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

# 3. 进行模型预测，并获取误分类的样本
incorrect_images = []
incorrect_labels = []
incorrect_preds = []

# 遍历测试数据集
for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)

    # 模型预测
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    # 找出误分类的样本
    incorrect_indices = torch.where(preds != labels)[0]

    for idx in incorrect_indices:
        incorrect_images.append(images[idx].cpu())
        incorrect_labels.append(labels[idx].cpu())
        incorrect_preds.append(preds[idx].cpu())

# 4. 展示误分类的样本
num_images = len(incorrect_images)
print(f"Total incorrect samples: {num_images}")

# 显示前5个误分类的样本
for i in range(min(num_images, 10)):
    img = incorrect_images[i].permute(1, 2, 0).numpy()  # 转换为 HxWxC 格式
    true_label = incorrect_labels[i].item()
    pred_label = incorrect_preds[i].item()

    # 显示图像
    plt.imshow(img)
    plt.title(
        f"True: {class_names[true_label]} | Pred: {class_names[pred_label]}")
    plt.axis('off')
    plt.savefig(f"./mistake/{i}.png")
    plt.show()
