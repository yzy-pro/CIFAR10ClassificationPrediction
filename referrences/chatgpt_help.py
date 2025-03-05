import os
import shutil

# def clear_model_folder(folder_path='model'):
#     # 检查文件夹是否存在
#     if os.path.exists(folder_path):
#         # 遍历文件夹中的所有内容并删除
#         for filename in os.listdir(folder_path):
#             file_path = os.path.join(folder_path, filename)
#             try:
#                 # 如果是文件，则删除
#                 if os.path.isfile(file_path):
#                     os.remove(file_path)
#                 # 如果是文件夹，则递归删除
#                 elif os.path.isdir(file_path):
#                     shutil.rmtree(file_path)
#             except Exception as e:
#                 print(f"Error removing {file_path}: {e}")
#     else:
#         print(f"Folder {folder_path} does not exist!")
#
# # 调用函数清空 model 文件夹
# clear_model_folder('models')

# import matplotlib.pyplot as plt
# import numpy as np
#
# # 生成数据
# x = np.linspace(0, 10, 100)
# y1 = np.sin(x)  # 第一条曲线数据
# y2 = np.cos(x)  # 第二条曲线数据
#
# # 创建图形
# fig, ax1 = plt.subplots()
#
# # 绘制第一条曲线，使用左侧坐标轴
# ax1.plot(x, y1, 'g-', label='sin(x)')
# ax1.set_xlabel('X')
# ax1.set_ylabel('sin(x)', color='g')
# ax1.tick_params(axis='y', labelcolor='g')
#
# # 创建右侧坐标轴
# ax2 = ax1.twinx()
#
# # 绘制第二条曲线，使用右侧坐标轴
# ax2.plot(x, y2, 'b-', label='cos(x)')
# ax2.set_ylabel('cos(x)', color='b')
# ax2.tick_params(axis='y', labelcolor='b')
#
# # 显示图形
# plt.title('Two Y axes with different scales')
# plt.show()

#
# directory = "models"
# pattern = re.compile(r'acc_([\d.]+)')
#
# best_acc = -1
# best_model = None
#
# for filename in os.listdir(directory):
#     match = pattern.search(filename)
#     if match:
#         acc_str = match.group(1).rstrip('.')
#         try:
#             acc = float(acc_str)
#             if acc > best_acc:
#                 best_acc = acc
#                 best_model = os.path.join(directory, filename)
#         except ValueError:
#             print(f"error: {acc_str} ({filename})")
#
# print(f"using model: {best_model}")