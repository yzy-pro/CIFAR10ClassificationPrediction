from matplotlib import pyplot as plt
import os
import torch


# 为了避免matplotlib无法直接处理gpu上的list
# 将其转换为cpu上的numpy
def to_numpy(data):
    if isinstance(data, list):
        data = torch.tensor(data)
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data


# 绘图函数
def loss_plt(train_loss, valid_loss, accuracy):

    # 左侧：Loss
    fig, plt1 = plt.subplots()
    plt1.plot(train_loss, label="train_loss")
    plt1.plot(valid_loss, label="valid_loss")
    plt1.set_ylabel("loss")
    plt1.legend()

    # 右侧：准确率
    plt2 = plt1.twinx()
    plt2.plot(accuracy, label="accuracy")
    plt2.set_ylabel("accuracy")
    plt2.legend()

    plt.title("Loss and Accuracy with Epoches")
    plt.xlabel("Epoch")

    os.makedirs(os.path.join(os.getcwd(), "plots"), exist_ok=True)
    plt.savefig(os.path.join(os.getcwd(), "plots/loss_and_acc.png"))

    plt.show()
