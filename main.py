from torch.utils.data import DataLoader

import data_loader
import module
import plot
import torch
import os


device = torch.device("xpu" if torch.xpu.is_available() else "cpu")

train_data, test_data = data_loader.data_loader()
train_data_load = DataLoader(dataset=train_data, batch_size=64,
                               shuffle=True, drop_last=True)
test_data_load = DataLoader(dataset=test_data, batch_size=64,
                              shuffle=True, drop_last=True)

mynet = module.mynet.to(device)
loss_fn = module.loss_fn.to(device)
optim = module.optim

if __name__ == "__main__":
    epochs = 2

    train_loss_plt = []
    test_loss_plt = []
    acc_plt = []

    for epoch in range(epochs):
        print("Epoch {}/{} ********************************"
              .format(epoch + 1, epochs))
        train_step = 0
        mynet.train()
        losses = []
        for j, (imgs, labels) in enumerate(train_data_load):
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = mynet(imgs)
            loss = loss_fn(outputs, labels)

            losses.append(loss)

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_step += 1
            if train_step % 100 == 0:
                print("train_round %d, loss: %f" % (train_step, loss.item()))

        train_loss_plt.append(sum(losses) / len(losses))

        mynet.eval()
        accuracy = 0
        total_accuracy = 0
        losses = []
        with torch.no_grad():
            for j, (imgs, labels) in enumerate(test_data_load):

                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = mynet(imgs)
                loss = loss_fn(outputs, labels)

                losses.append(loss)

                labels = labels.to(outputs.device)

                # print(f"Outputs device: {outputs.device}")
                # print(f"Labels device: {labels.device}")
                # print(f"Outputs type: {type(outputs)}")
                # print(f"Outputs shape: {outputs.shape}")
                # print(f"Labels shape: {labels.shape}")

                accuracy = (outputs.argmax(axis=1).cpu() == labels.cpu()).sum(

                ).item()
                total_accuracy += accuracy
                average_accuracy = total_accuracy / (len(test_data_load) * 64)

            test_loss_plt.append(sum(losses) / len(losses))
            acc_plt.append(average_accuracy)

            print("Epoch %d, accuracy: %f" % (epoch + 1,
                                                   average_accuracy))
            os.makedirs(os.path.join(os.getcwd(), 'models'), exist_ok=True)
            torch.save(mynet, f'./models/E_{epoch + 1}_acc_{average_accuracy}.pth')

    train_loss_plt, test_loss_plt, acc_plt =\
        plot.to_numpy(train_loss_plt), plot.to_numpy(test_loss_plt), plot.to_numpy(acc_plt)

    plot.loss_plt(train_loss_plt, test_loss_plt, acc_plt)