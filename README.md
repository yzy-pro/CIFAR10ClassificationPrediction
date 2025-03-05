# 这个旧版本已经放弃继续研究了
# Comsen
## 华中科技大学光学与电子信息学院2409陈恪瑾
### 项目概述
#### 项目名称
CIFAR分类
#### 数据来源
torch.datasets.CIFAR10
#### 使用模型
基于pytroch的LeNet

[model.py](v_1/model.py)
#### requirements
[requirements.txt](v_1/requirements.txt)

* conda install libuv
python -m pip install torch==2.5.1.post0+cxx11.abi torchvision==0.20.1.post0+cxx11.abi torchaudio==2.5.1.post0+cxx11.abi intel-extension-for-pytorch==2.5.10.post0+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/arl/us/

#### 主要参考文献（代码参考）
张奥森, Zachary C. Lipton, 李沐, Alexander J. Smola. 动手学深度学习[M]. 机械工业出版社, 2021.

[lenet.ipynb](referrences/lenet.ipynb)

https://www.cnblogs.com/xiaoxing-chen/p/18537910
### 项目成果
#### 源代码
训练代码：
[model_train.py](v_1/model_train.py)

测试代码：
[model_test.py](v_1/model_test.py)
#### 训练数据
[cifar-10-batches-py](data/test/cifar-10-batches-py)

[cifar-10-batches-py](data/train/cifar-10-batches-py)
#### 预测结果
[model_test.py](v_1/model_test.py)
#### 导出模型
[best_model.pth](v_1/module/best_model.pth)
#### 训练可视化曲线图
[plots](v_1/plots)


 