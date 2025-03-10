# 使用说明：
# 此程序将会自动使用models文件夹下的准确率最高的模型

# 将待预测的图片置入 app 文件夹下，运行程序，即可得到预测结果

import os
import torch
from PIL import Image
import torchvision
import re

labels_idx = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

directory = "models"
pattern = re.compile(r"acc_([\d.]+)")

best_acc = -1
best_model = None

for filename in os.listdir(directory):
    match = pattern.search(filename)
    if match:
        acc_str = match.group(1).rstrip(".")
        try:
            acc = float(acc_str)
            if acc > best_acc:
                best_acc = acc
                best_model = os.path.join(directory, filename)
        except ValueError:
            print(f"error: {acc_str} ({filename})")

print(f"using model: {best_model}")
app_net = torch.load(best_model, map_location=torch.device("cpu"))

image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
image_files = [
    file for file in os.listdir("app") if file.lower().endswith(image_extensions)
]

for app_file in image_files:
    app_path = os.path.join("./app", app_file)
    app_img = Image.open(app_path).convert("RGB")
    # app_img.show()
    tran_pose = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=(32, 32)),
            torchvision.transforms.ToTensor(),
        ]
    )

    app_img = tran_pose(app_img)
    app_img = torch.reshape(app_img, (1, 3, 32, 32))
    app_output = app_net(app_img)
    answer = labels_idx[app_output.argmax(axis=1).item()]
    print(f"I think {app_file} is a/an {answer}")
