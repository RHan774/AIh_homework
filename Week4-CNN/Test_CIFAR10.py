import my_utils
import numpy as np
from Model import CNN

# 从config文件中读取配置的参数
dataset = 'CIFAR10'
config = my_utils.read_config(dataset)
config_test = config['Test']

# 测试参数
batch_size = config_test['batch_size']
model_path = config_test['model_path']

# 数据路径
test_datapath = config_test['data_path']
test_labelpath = config_test['label_path']

# 模型参数
activation = config_test['activation_function']
conv_channels = config_test['conv_channels']
conv_kernel_sizes = config_test['conv_kernel_sizes']
conv_strides = config_test['conv_strides']
conv_paddings = config_test['conv_paddings']
pool_sizes = config_test['pool_sizes']
pool_strides = config_test['pool_strides']
fc_sizes = config_test['fc_sizes']

if __name__ == "__main__":
    
    # 根据config中参数定义激活函数
    if activation == 0:
        activation_function = my_utils.sigmoid
        activation_derivation = my_utils.sigmoid_derivation
    elif activation == 1:
        activation_function = my_utils.relu
        activation_derivation = my_utils.relu_derivation
    elif activation == 2:
        activation_function = my_utils.tanh
        activation_derivation = my_utils.tanh_derivation
    elif activation == 3:
        activation_function = my_utils.leaky_relu
        activation_derivation = my_utils.leaky_relu_derivation
    elif activation == 4:
        activation_function = my_utils.prelu
        activation_derivation = my_utils.prelu_derivation

    # 加载测试集数据
    try:
        test_data = np.load(test_datapath)
        test_labels = np.load(test_labelpath)
        print(f"测试集: {test_data.shape}, {test_labels.shape}")
    except FileNotFoundError:
        print("测试数据文件不存在，请先运行 data_process.py 处理数据")
        exit()

    # 初始化CNN模型
    cifar_cnn = CNN(
        batch_size=batch_size,
        conv_channels=conv_channels,
        conv_kernel_sizes=conv_kernel_sizes,
        conv_strides=conv_strides,
        conv_paddings=conv_paddings,
        pool_sizes=pool_sizes,
        pool_strides=pool_strides,
        fc_sizes=fc_sizes,
        activation_function=activation_function,
        activation_derivation=activation_derivation,
        is_load=True,
        model_path=model_path
    )
    
    # 测试模型
    test_loss, test_accuracy = cifar_cnn.val(test_data, test_labels)
    print(f"CIFAR-10测试集: 准确率 = {test_accuracy:.4f}, 损失 = {test_loss:.6f}") 