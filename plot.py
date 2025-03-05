from matplotlib import pyplot as plt
import os
import torch

def to_numpy(data):
    if isinstance(data, list):
        data = torch.tensor(data)
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data

def loss_plt(train_loss, valid_loss, accuracy):

    fig, plt1 = plt.subplots()
    plt1.plot(train_loss, label='train_loss')
    plt1.plot(valid_loss, label='valid_loss')
    plt1.legend()

    plt2 = plt1.twinx()
    plt2.plot(accuracy, label='accuracy')
    plt2.legend()

    plt.title('Loss and Accuracy with Epoches')

    os.makedirs(os.path.join(os.getcwd(), 'plots'), exist_ok=True)
    plt.savefig(os.path.join(os.getcwd(), 'plots/loss_and_acc.png'))

    plt.show()