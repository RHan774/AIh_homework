# CNN 卷积神经网络实现

本项目基于NumPy实现了卷积神经网络（CNN），用于MNIST和CIFAR-10数据集的图像分类任务。

## 项目结构

- `Model.py`: CNN模型定义，包含卷积层、池化层、Flatten层、Dropout层、全连接层等
- `my_utils.py`: 工具函数，包含激活函数、读取配置等
- `config.yaml`: 配置文件，包含模型参数、训练参数等
- `data_process.py`: 数据处理脚本，用于读取和处理MNIST和CIFAR-10数据集
- `Train_MNIST.py`: MNIST数据集训练脚本
- `Train_CIFAR10.py`: CIFAR-10数据集训练脚本
- `Test_MNIST.py`: MNIST数据集测试脚本
- `Test_CIFAR10.py`: CIFAR-10数据集测试脚本

## 使用方法

### 1. 准备数据集

#### MNIST数据集

MNIST数据集应放在`./dataset/MNIST/`目录下，包含以下文件：
- `train-images.idx3-ubyte`: 训练图像
- `train-labels.idx1-ubyte`: 训练标签
- `test-images.idx3-ubyte`: 测试图像
- `test-labels.idx1-ubyte`: 测试标签

#### CIFAR-10数据集

CIFAR-10数据集应放在`./dataset/cifar-10-batches-py/`目录下，包含以下文件：
- `data_batch_1`, `data_batch_2`, `data_batch_3`, `data_batch_4`, `data_batch_5`: 训练数据
- `test_batch`: 测试数据
- `batches.meta`: 元数据

### 2. 数据预处理

运行以下命令处理数据集：

```bash
python data_process.py
```

这将会读取原始数据集，并生成处理后的numpy数组文件：
- `./dataset/MNIST/train_data.npy`
- `./dataset/MNIST/train_label.npy`
- `./dataset/MNIST/val_data.npy`
- `./dataset/MNIST/val_label.npy`
- `./dataset/MNIST/test_data.npy`
- `./dataset/MNIST/test_label.npy`
- `./dataset/CIFAR10/train_data.npy`
- `./dataset/CIFAR10/train_label.npy`
- `./dataset/CIFAR10/val_data.npy`
- `./dataset/CIFAR10/val_label.npy`
- `./dataset/CIFAR10/test_data.npy`
- `./dataset/CIFAR10/test_label.npy`

### 3. 修改配置

根据需要修改`config.yaml`文件中的参数，例如：
- 调整卷积层通道数：`conv_channels`
- 调整全连接层大小：`fc_sizes`
- 调整学习率：`learning_rate`
- 调整批大小：`batch_size`
- 调整训练轮数：`epoches`
- 调整是否使用学习率调度等

### 4. 训练模型

#### 训练MNIST模型

```bash
python Train_MNIST.py
```

#### 训练CIFAR-10模型

```bash
python Train_CIFAR10.py
```

训练过程中，每5个epoch会评估一次模型性能，并保存最佳模型。

### 5. 测试模型

#### 测试MNIST模型

```bash
python Test_MNIST.py
```

#### 测试CIFAR-10模型

```bash
python Test_CIFAR10.py
```

## 模型架构

### MNIST模型架构

- 输入：28x28灰度图像（1通道）
- 卷积层1：16个5x5卷积核，步长1
- 激活函数：ReLU
- 最大池化层1：2x2池化核，步长2
- 卷积层2：32个5x5卷积核，步长1
- 激活函数：ReLU
- 最大池化层2：2x2池化核，步长2
- Flatten层
- 全连接层1：512个神经元
- 激活函数：ReLU
- Dropout：0.5
- 全连接层2（输出层）：10个神经元（对应10个类别）
- Softmax激活函数

### CIFAR-10模型架构

- 输入：32x32彩色图像（3通道）
- 卷积层1：32个3x3卷积核，步长1
- 激活函数：ReLU
- 最大池化层1：2x2池化核，步长2
- 卷积层2：64个3x3卷积核，步长1
- 激活函数：ReLU
- 最大池化层2：2x2池化核，步长2
- 卷积层3：128个3x3卷积核，步长1
- 激活函数：ReLU
- 最大池化层3：2x2池化核，步长2
- Flatten层
- 全连接层1：512个神经元
- 激活函数：ReLU
- Dropout：0.5
- 全连接层2（输出层）：10个神经元（对应10个类别）
- Softmax激活函数 