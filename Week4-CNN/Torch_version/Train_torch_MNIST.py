import torch.nn as nn
import os
from Model_torch import CNNTorch
import my_utils

# 设置环境变量，解决OpenMP警告
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


if __name__ == "__main__":
    dataset = 'MNIST'
    config = my_utils.read_config(dataset)
    config_train = config['Train']
    config_val = config['Val']
    
    # 获取CNN架构参数
    conv_channels = config_train['conv_channels']  # 包含输入通道数
    conv_kernel_sizes = config_train['conv_kernel_sizes']
    conv_strides = config_train['conv_strides']
    conv_paddings = config_train['conv_paddings']
    pool_sizes = config_train['pool_sizes']
    pool_strides = config_train['pool_strides']
    fc_sizes = config_train['fc_sizes']
    use_dropout = config_train['use_dropout']
    dropout_rate = config_train['dropout_rates'][0] if len(config_train['dropout_rates']) > 0 else 0.0
    
    # 初始化CNN模型
    model = CNNTorch(
        conv_channels=conv_channels,
        conv_kernel_sizes=conv_kernel_sizes,
        conv_strides=conv_strides,
        conv_paddings=conv_paddings,
        pool_sizes=pool_sizes,
        pool_strides=pool_strides,
        fc_sizes=fc_sizes,
        dropout_rate=dropout_rate,
        use_dropout=use_dropout
    )
    
    # 调用训练函数
    my_utils.train(model, config_train, config_val) 