# 数据下载核导入模块
import torchvision


def data_loader():
    train_data_set = torchvision.datasets.CIFAR10(
        root="./data/train",
        train=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(
                    size=(32, 32), padding=4
                ),  # 随机裁剪
                torchvision.transforms.RandomHorizontalFlip(p=0.5),  # 随机翻转
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
                # 标准化
            ]
        ),
        download=True,
    )

    test_data_set = torchvision.datasets.CIFAR10(
        root="./data/test",
        train=False,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        ),
        download=True,
    )

    return train_data_set, test_data_set


if __name__ == "__main__":
    train_data_set, test_data_set = data_loader()
    print(f"size of train_data:{len(train_data_set)}")
    print(f"size of test_data:{len(test_data_set)}")
