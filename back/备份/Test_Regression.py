import my_utils
import numpy as np
from Modle import MLP

# 从config文件中读取配置的参数
task = 'Regression'
is_load = True
config = my_utils.read_config(task)
config_train = config['Train']
config_test = config['Test']

batch_size = config_train['batch_size']
epoches = config_train['epoches']
layer_arch = config_train['layer_arch']
lr = config_train['learning_rate']
data_size = config_train['data_size']
random_range = config_train['init_params_random_range']
activation = config_train['activation_function']
load_path = config_train['model_path']
x_path = config_test['x_data']
y_path = config_test['y_data']

if __name__ == "__main__":
    
    # 根据config中参数定义激活函数
    if activation == 0:
        activation_function = my_utils.sigmoid
        activation_derivation = my_utils.sigmoid_derivation
    elif activation == 1:
        activation_function = my_utils.relu
        activation_derivation = my_utils.relu_derivation
    else:
        activation_function = my_utils.tanh
        activation_derivation = my_utils.tanh_derivation

    # 加载训练集和验证集的数据
    x_data = np.load(x_path)
    y_data = np.load(y_path)

    # 初始化模型
    ClassifierBP  = MLP(task, layer_arch, lr, batch_size, random_range,
                        activation_function, activation_derivation, is_load, load_path)

    loss = ClassifierBP.val(x_data, y_data)
    print("Test loss =", loss)