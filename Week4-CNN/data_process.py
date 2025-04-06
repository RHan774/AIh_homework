import pickle
import numpy as np
import os
import struct
import my_utils
import matplotlib.pyplot as plt

# 数据地址
images_addr = './Week4-CNN/dataset/cifar-10-batches-py/'

def unpickle(file):
    """
    读取CIFAR-10的二进制文件
    
    Args:
        file: 文件路径
        
    Returns:
        解析后的数据字典
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_mnist(images_addr, labels_addr, num_samples, one_hot=True, flatten=False):
    """
    读取MNIST数据集
    
    Args:
        images_addr: 图像文件地址
        labels_addr: 标签文件地址
        num_samples: 读取的样本数量
        one_hot: 是否将标签转为one-hot编码
        flatten: 是否将图像展平为1维向量
        
    Returns:
        图像数据和标签
    """
    # 读取图像文件
    with open(images_addr, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        # 读取指定数量的图像
        n = min(n, num_samples)
        images = np.fromfile(f, dtype=np.uint8).reshape(n, rows, cols)
        
    # 读取标签文件
    with open(labels_addr, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        # 读取指定数量的标签
        n = min(n, num_samples)
        labels = np.fromfile(f, dtype=np.uint8).reshape(n)
    
    # 将图像数据标准化到[0, 1]
    images = images.astype(np.float32) / 255.0
    # 将图像数据增加一个维度（对应CIFAR10的3通道）
    images = images.reshape(num_samples, 28, 28, 1)
    
    # 标签转为one-hot编码
    if one_hot:
        # 创建一个n x 10的零矩阵
        one_hot_labels = np.zeros((n, 10))
        # 使用索引和标签值设置对应位置为1
        one_hot_labels[np.arange(n), labels] = 1
        return images, one_hot_labels
    
    return images, labels

def load_cifar10(data_path, num_samples=None, one_hot=True, train=True):
    """
    读取CIFAR-10数据集
    
    Args:
        data_path: 数据目录路径
        num_samples: 读取的样本数量，None表示读取所有数据
        one_hot: 是否将标签转为one-hot编码
        train: 是否读取训练集，False表示读取测试集
        
    Returns:
        图像数据和标签
    """
    if train:
        # 读取训练数据
        batch_files = [
            f"{data_path}/data_batch_1",
            f"{data_path}/data_batch_2",
            f"{data_path}/data_batch_3",
            f"{data_path}/data_batch_4",
            f"{data_path}/data_batch_5",
        ]
        
        # 读取所有批次的数据
        data_batches = [unpickle(file) for file in batch_files]
        
        # 合并所有批次的数据
        data = np.vstack([batch[b'data'] for batch in data_batches])
        labels = np.hstack([batch[b'labels'] for batch in data_batches])
    else:
        # 读取测试数据
        test_batch = unpickle(f"{data_path}/test_batch")
        data = test_batch[b'data']
        labels = np.array(test_batch[b'labels'])
    
    # 限制样本数量
    if num_samples is not None:
        num_samples = min(num_samples, len(data))
        data = data[:num_samples]
        labels = labels[:num_samples]
    
    # 重塑数据为图像格式 (N, 3, 32, 32) -> (N, 32, 32, 3)
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    # 将图像数据标准化到[0, 1]
    data = data.astype(np.float32) / 255.0
    
    # 标签转为one-hot编码
    if one_hot:
        # 创建一个n x 10的零矩阵
        one_hot_labels = np.zeros((len(labels), 10))
        # 使用索引和标签值设置对应位置为1
        one_hot_labels[np.arange(len(labels)), labels] = 1
        return data, one_hot_labels
    
    return data, labels

def save_mnist_data(flatten=False, one_hot=True):
    """
    处理MNIST数据集并保存到指定目录
    
    Args:
        flatten: 是否将图像展平为1维向量
        one_hot: 是否将标签转为one-hot编码
    """
    # 从config文件读取配置
    config = my_utils.read_config('DataProcess')['MNIST']
    
    # 数据集路径
    images_addr = config.images_addr
    labels_addr = config.labels_addr
    test_images = config.test_images
    test_labels = config.test_labels
    
    # 样本数量
    train_num = config.train_num
    validate_num = config.validate_num
    test_num = config.test_num
    
    # 保存路径
    data_path = config.data_path
    
    # 创建保存目录
    os.makedirs(data_path, exist_ok=True)
    
    # 读取所有训练数据
    all_train_images, all_train_labels = load_mnist(images_addr, labels_addr, train_num + validate_num, one_hot, flatten)
    
    # 分割训练集和验证集
    train_data = all_train_images[:train_num]
    train_label = all_train_labels[:train_num]
    val_data = all_train_images[train_num:train_num + validate_num]
    val_label = all_train_labels[train_num:train_num + validate_num]
    
    # 读取测试数据
    test_data, test_label = load_mnist(test_images, test_labels, test_num, one_hot, flatten)
    
    # 保存数据
    np.save(f"{data_path}/train_data.npy", train_data)
    np.save(f"{data_path}/train_label.npy", train_label)
    np.save(f"{data_path}/val_data.npy", val_data)
    np.save(f"{data_path}/val_label.npy", val_label)
    np.save(f"{data_path}/test_data.npy", test_data)
    np.save(f"{data_path}/test_label.npy", test_label)
    
    print(f"训练集: {train_data.shape}, {train_label.shape}")
    print(f"验证集: {val_data.shape}, {val_label.shape}")
    print(f"测试集: {test_data.shape}, {test_label.shape}")
    print(f"MNIST数据保存到: {data_path}")

def save_cifar10_data(one_hot=True):
    """
    处理CIFAR-10数据集并保存到指定目录
    
    Args:
        one_hot: 是否将标签转为one-hot编码
    """
    # 从config文件读取配置
    config = my_utils.read_config('DataProcess')['CIFAR10']
    
    # 数据集路径
    data_path = config.data_path
    
    # 样本数量
    train_num = config.train_num
    validate_num = config.validate_num
    test_num = config.test_num
    
    # 保存路径
    save_path = "./dataset/CIFAR10/"
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 读取所有训练数据
    all_train_data, all_train_labels = load_cifar10(data_path, None, one_hot, train=True)
    
    # 分割训练集和验证集
    train_data = all_train_data[:train_num]
    train_label = all_train_labels[:train_num]
    val_data = all_train_data[train_num:train_num + validate_num]
    val_label = all_train_labels[train_num:train_num + validate_num]
    
    # 读取测试数据
    test_data, test_label = load_cifar10(data_path, test_num, one_hot, train=False)
    
    # 保存数据
    np.save(f"{save_path}/train_data.npy", train_data)
    np.save(f"{save_path}/train_label.npy", train_label)
    np.save(f"{save_path}/val_data.npy", val_data)
    np.save(f"{save_path}/val_label.npy", val_label)
    np.save(f"{save_path}/test_data.npy", test_data)
    np.save(f"{save_path}/test_label.npy", test_label)
    
    print(f"训练集: {train_data.shape}, {train_label.shape}")
    print(f"验证集: {val_data.shape}, {val_label.shape}")
    print(f"测试集: {test_data.shape}, {test_label.shape}")
    print(f"CIFAR-10数据保存到: {save_path}")

if __name__ == "__main__":
    # 处理并保存MNIST数据集
    print("处理MNIST数据集...")
    save_mnist_data(flatten=False, one_hot=True)
    
    # 处理并保存CIFAR-10数据集
    print("\n处理CIFAR-10数据集...")
    save_cifar10_data(one_hot=True)

# # 读取元数据获取类别名称
# meta_data = unpickle('batches.meta')
# label_names = [label.decode('utf-8') for label in meta_data[b'label_names']]

# # 显示一些图像示例
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.imshow(train_data[i])
#     plt.title(label_names[train_labels[i]])
#     plt.axis('off')
# plt.tight_layout()
# plt.show()