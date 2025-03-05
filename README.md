# Comsen
## 华中科技大学光学与电子信息学院2409陈恪瑾
### 项目概述
#### 项目名称
CIFAR分类
#### 数据来源
torch.datasets.CIFAR10
#### 使用模型
基于pytroch的LeNet

#### requirements
[requirements.txt](requirements.txt)

##### 关于安装intel的gpu加速
* conda install libuv
python -m pip install torch==2.5.1.post0+cxx11.abi torchvision==0.20.1.post0+cxx11.abi torchaudio==2.5.1.post0+cxx11.abi intel-extension-for-pytorch==2.5.10.post0+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/arl/us/

#### 主要参考文献（代码参考）
* 张奥森, Zachary C. Lipton, 李沐, Alexander J. Smola. 动手学深度学习[M]. 机械工业出版社, 2021.

* [lenet.ipynb](referrences/lenet.ipynb)

* https://www.cnblogs.com/xiaoxing-chen/p/18537910
### 项目成果
* 最终准确率：80% ~ 85%
* 使用项目成果，请运行[application.py](application.py)
#### 源代码
* 加载数据[data_loader.py](data_loader.py)
* 建立模型[module.py](module.py)
* 训练[main.py](main.py)
* 展示训练效果[plot.py](plot.py)
* 实际应用[application.py](application.py)
#### 训练数据
* 训练集[cifar-10-python.tar.gz](data/train/cifar-10-python.tar.gz)
* 测试集[cifar-10-python.tar.gz](data/test/cifar-10-python.tar.gz)

#### 导出模型
* [models](models)
#### 训练可视化曲线图
* [plots](plots)


 