import numpy as np
import argparse
import yaml
from easydict import EasyDict
import math

def read_config(task_kind):
    parser = argparse.ArgumentParser(description = task_kind)
    parser.add_argument("--config_path", type=str, default="config.yaml")
    args = parser.parse_args()
    config_path =args.config_path
    config = yaml.load(open(config_path, 'r', encoding='utf-8'), Loader=yaml.Loader)
    config = EasyDict(config)
    # 设置随机种子
    if 'Global' in config:
        np.random.seed(config.Global.random_seed)
    config = config[task_kind]
    return config

def get_lr_by_scheduler(base_lr, current_epoch, total_epochs, warmup_epochs=0, use_warmup=False, use_cosine_decay=False, min_lr_ratio=0.01):
    """计算当前epoch的学习率
    
    Args:
        base_lr: 基础学习率（初始学习率）
        current_epoch: 当前的epoch数
        total_epochs: 总的训练epoch数
        warmup_epochs: 预热的epoch数
        use_warmup: 是否使用学习率预热
        use_cosine_decay: 是否使用余弦衰减
        min_lr_ratio: 最小学习率与基础学习率的比例
        
    Returns:
        当前epoch的学习率
    """
    min_lr = base_lr * min_lr_ratio
    
    # 学习率预热阶段
    if use_warmup and current_epoch < warmup_epochs:
        # 线性预热：从 min_lr 到 base_lr
        return min_lr + (base_lr - min_lr) * current_epoch / warmup_epochs
    
    # 学习率衰减阶段
    if use_cosine_decay:
        # 考虑预热阶段后的衰减
        if use_warmup:
            # 预热后的epoch数
            after_warmup_epoch = current_epoch - warmup_epochs
            # 预热后的总epoch数
            after_warmup_total = total_epochs - warmup_epochs
            # 避免除以0
            if after_warmup_total <= 0:
                return base_lr
            # 余弦衰减，从base_lr到min_lr
            cosine_decay = 0.5 * (1 + math.cos(math.pi * after_warmup_epoch / after_warmup_total))
            return min_lr + (base_lr - min_lr) * cosine_decay
        else:
            # 没有预热，直接从头开始余弦衰减
            cosine_decay = 0.5 * (1 + math.cos(math.pi * current_epoch / total_epochs))
            return min_lr + (base_lr - min_lr) * cosine_decay
    
    # 不使用任何调度，返回基础学习率
    return base_lr

'''sigmoid_原->sigmoid_2和sigmoid'''
# def sigmoid(x):
#     # try:
#     #     exp = np.exp(-x)
#     # except OverflowError as e:
#     #         print("数值溢出错误：", x)
#     return 1. / (1 + np.exp(-x))

'''sigmoid_new -> sigmoid_new'''
def sigmoid(x):
    # 解决溢出问题
    # 把大于0和小于0的元素分别处理
    # 原来的sigmoid函数是 1/(1+np.exp(-x))
    # 当x是比较小的负数时会出现上溢，此时可以通过计算exp(x) / (1+exp(x)) 来解决
    
    mask = (x > 0)
    positive_out = np.zeros_like(x, dtype='float64')
    negative_out = np.zeros_like(x, dtype='float64')
    
    # 大于0的情况
    positive_out = 1 / (1 + np.exp(-x, positive_out, where=mask))
    # 清除对小于等于0元素的影响
    positive_out[~mask] = 0
    
    # 小于等于0的情况
    expZ = np.exp(x,negative_out,where=~mask)
    negative_out = expZ / (1+expZ)
    # 清除对大于0元素的影响
    negative_out[mask] = 0
    
    return positive_out + negative_out

'''sigmoid_nnew -> sigmoid_nnew'''
# def sigmoid(x):
#     indices_pos = np.nonzero(x>=0)
#     indices_neg = np.nonzero(x<0)
#     n_rows, n_cols = np.shape(x)
#     result = np.zeros((n_rows, n_cols))
#     result[indices_pos] = 1 / (1 + np.exp(-x[indices_pos]))
#     neg_exp = np.exp(x[indices_neg])
#     result[indices_neg] = neg_exp / (1 + neg_exp)
#     return result

def sigmoid_derivation(x):
    return x * (1- x)

def relu(x):
    return np.maximum(x,0)

def relu_derivation(x):
    grad = np.array(x,copy=True)
    grad[x > 0] = 1
    grad[x <= 0] = 0
    return grad
    
def tanh(x):
    return np.tanh(x)

def tanh_derivation(x):
    return 1 - x ** 2

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU激活函数"""
    return np.maximum(alpha * x, x)

def leaky_relu_derivation(x, alpha=0.01):
    """Leaky ReLU激活函数的导数"""
    grad = np.ones_like(x)
    grad[x < 0] = alpha
    return grad

def prelu(x, alpha=0.01):
    """PReLU激活函数 (参数化的Leaky ReLU)"""
    return np.maximum(alpha * x, x)

def prelu_derivation(x, alpha=0.01):
    """PReLU激活函数的导数"""
    grad = np.ones_like(x)
    grad[x < 0] = alpha
    return grad

def gelu(x):
    """GELU (Gaussian Error Linear Unit) 激活函数"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def gelu_derivation(x):
    """GELU激活函数的导数"""
    # 近似导数计算
    cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    pdf = np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)
    return cdf + x * pdf

def swish(x, beta=1.0):
    """Swish激活函数: x * sigmoid(beta * x)"""
    return x * sigmoid(beta * x)

def swish_derivation(x, beta=1.0):
    """Swish激活函数的导数"""
    sig = sigmoid(beta * x)
    return sig + x * beta * sig * (1 - sig)

def swiglu(x):
    """SwiGLU激活函数 (Swish-Gated Linear Unit)
    注意：SwiGLU通常输入分为两部分，这里简化为一个实现"""
    half = x.shape[1] // 2
    x1, x2 = x[:, :half], x[:, half:]
    return swish(x1) * x2

def swiglu_derivation(x):
    """SwiGLU激活函数导数"""
    half = x.shape[1] // 2
    x1, x2 = x[:, :half], x[:, half:]
    
    # 计算swish(x1)的导数
    swish_deriv = swish_derivation(x1)
    
    # 创建结果数组
    result = np.zeros_like(x)
    result[:, :half] = swish_deriv * x2  # 对x1的导数
    result[:, half:] = swish(x1)         # 对x2的导数
    
    return result

def softmax(x):
    exp_x = np.exp(x)
    #exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))    # 减去最大值避免数值溢出
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def get_batch(data, labels, batch_size, shuffle=True):
    """
    将数据划分为指定大小的batch
    
    Args:
        data: 输入数据，形状为 (样本数, 特征维度)
        labels: 标签数据，形状为 (样本数, 标签维度)
        batch_size: 每个batch的样本数量
        shuffle: 是否打乱数据. default to True
        
    Returns:
        批次数据生成器
    """
    # 获取数据样本总数
    data_size = len(data)
    
    # 计算可以获得的batch数
    n_batches = data_size // batch_size
    
    # 生成索引数组
    indices = np.arange(data_size)
    
    # 如果需要打乱，则随机打乱索引
    if shuffle:
        np.random.shuffle(indices)
    
    # 返回每个batch的数据
    for i in range(n_batches):
        # 当前batch的索引
        batch_indices = indices[i * batch_size:(i + 1) * batch_size]
        
        # 根据索引获取当前batch的数据和标签
        batch_data = data[batch_indices].reshape(batch_size, -1)
        batch_labels = labels[batch_indices].reshape(batch_size, -1)
        
        yield batch_data, batch_labels


def generate_sin_data(train_num, validate_num, test_num, noise_std, data_path):
    """生成拟合sin函数的训练及测试数据

    Args:
        train_num : 训练样本数.
        validate_num : 验证样本数.
        test_num : 测试样本数.
        noise_std : 噪声标准差.
        data_path : 存放数据集路径. 
    """
    x_range = (-np.pi, np.pi)   # x取值范围为[-pi, pi]
    
    # 生成训练集
    x_train = np.random.uniform(x_range[0], x_range[1], train_num)
    y_train = np.sin(x_train) + np.random.normal(0., noise_std, train_num)
    
    # 生成验证集
    x_validate = np.random.uniform(x_range[0], x_range[1], validate_num)
    y_validate = np.sin(x_validate) + np.random.normal(0., noise_std, validate_num)
    
    # 生成测试集
    x_test = np.random.uniform(x_range[0], x_range[1], test_num)
    y_test = np.sin(x_test) + np.random.normal(0., noise_std, test_num)
    
    # 保存为npy文件
    np.save(data_path + 'train_data_x.npy', x_train)
    np.save(data_path + 'train_data_y.npy', y_train)
    np.save(data_path + 'validate_data_x.npy', x_validate)
    np.save(data_path + 'validate_data_y.npy', y_validate)
    np.save(data_path + 'test_data_x.npy', x_test)
    np.save(data_path + 'test_data_y.npy', y_test)
    
def parse_train_data(images_addr, labels_addr, train_num, validate_num, flatten, one_hot):
    """解析train_data中的二进制文件, 并返回解析结果
    
    Args:
        images_addr: 图像数据集的文件地址.
        labels_addr: 标签数据集的文件地址.
        train_num: 训练集大小.
        validate_num: 验证集大小.
        flatten:  是否将图片展开, 即(n张, n_rows, n_cols)变成(n张, n_rows*n_cols).
        one_hot: 标签是否采用one hot形式.

    Returns:
        解析后的numpy数组
    """
    # 读取图像文件
    with open(images_addr, 'rb') as f:
        # 读取文件头信息
        magic = int.from_bytes(f.read(4), 'big')
        n_images = int.from_bytes(f.read(4), 'big')
        n_rows = int.from_bytes(f.read(4), 'big')
        n_cols = int.from_bytes(f.read(4), 'big')
        
        print("图片数：", n_images,"行数：", n_rows, "列数：", n_cols)
        
        # 读取图像数据
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(n_images, n_rows, n_cols)
        
    # 读取标签文件
    with open(labels_addr, 'rb') as f:
        # 读取文件头信息
        magic = int.from_bytes(f.read(4), 'big')
        n_labels = int.from_bytes(f.read(4), 'big')
        
        print("标签数：", n_labels)
        
        # 读取标签数据
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    # # 随机打乱数据
    # indices = np.random.permutation(n_images)
    # images = images[indices]
    # labels = labels[indices]
    
    # 划分训练集和验证集
    train_data = images[:train_num]
    train_label = labels[:train_num]
    validate_data = images[train_num:train_num + validate_num]
    validate_label = labels[train_num:train_num + validate_num]
    
    # 如果需要将图像展平
    if flatten:
        train_data = train_data.reshape(-1, n_rows*n_cols)
        validate_data = validate_data.reshape(-1, n_rows*n_cols)
    
    # 如果需要one-hot编码
    if one_hot:
        train_label = np.eye(10)[train_label]
        validate_label = np.eye(10)[validate_label]
    
    return train_data, train_label, validate_data, validate_label

def save_parse_data(images_addr, labels_addr, train_num, validate_num, flatten, one_hot, data_path):
    """保存划分的测试集与验证集
    
        Args:
        images_addr: 图像数据集的文件地址.
        labels_addr: 标签数据集的文件地址.
        train_num: 训练集大小.
        validate_num: 验证集大小.
        flatten:  是否将图片展开, 即(n张, 28, 28)变成(n张, 784).
        one_hot: 标签是否采用one hot形式.
    """
    # 解析数据集
    train_data, train_label, validate_data, validate_label = \
    parse_train_data(images_addr, labels_addr, train_num, validate_num, flatten, one_hot)
    # 保存为npy文件
    np.save(data_path + 'train_data.npy', train_data)
    np.save(data_path + 'train_label.npy', train_label)
    np.save(data_path + 'validate_data.npy', validate_data)
    np.save(data_path + 'validate_label.npy', validate_label)

def parse_test_data(images_addr, labels_addr, test_num, flatten, one_hot):
    # 读取图像文件
    with open(images_addr, 'rb') as f:
        # 读取文件头信息
        magic = int.from_bytes(f.read(4), 'big')
        n_images = int.from_bytes(f.read(4), 'big')
        n_rows = int.from_bytes(f.read(4), 'big')
        n_cols = int.from_bytes(f.read(4), 'big')
        
        print("图片数：", n_images,"行数：", n_rows, "列数：", n_cols)
        
        # 读取图像数据
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(n_images, n_rows, n_cols)
        
    # 读取标签文件
    with open(labels_addr, 'rb') as f:
        # 读取文件头信息
        magic = int.from_bytes(f.read(4), 'big')
        n_labels = int.from_bytes(f.read(4), 'big')
        
        print("标签数：", n_labels)
        
        # 读取标签数据
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    # # 随机打乱数据
    # indices = np.random.permutation(n_images)
    # images = images[indices]
    # labels = labels[indices]
    
    # 取出测试集（通常就是所有）
    test_data = images[:test_num]
    test_label = labels[:test_num]
    
    # 如果需要将图像展平
    if flatten:
        test_data = test_data.reshape(-1, n_rows*n_cols)
    
    # 如果需要one-hot编码
    if one_hot:
        test_label = np.eye(10)[test_label]
    
    return test_data, test_label

def save_test_data(images_addr, labels_addr, test_num, flatten, one_hot, data_path):
    test_data, test_label = parse_test_data(images_addr, labels_addr, test_num, flatten, one_hot)
    # 保存为npy文件
    np.save(data_path + 'test_data.npy', test_data)
    np.save(data_path + 'test_label.npy', test_label)