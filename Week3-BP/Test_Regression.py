import my_utils
import numpy as np
from Modle import MLP

# 从config文件中读取配置的参数
task = 'Regression'
is_load = True
config = my_utils.read_config(task)
config_test = config['Test']

x_path = config_test['x_data']
y_path = config_test['y_data']
load_path = config_test['model_path']
layer_arch = config_test['layer_arch']
batch_size = config_test['batch_size']
activation = config_test['activation_function']


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
    RegressionBP  = MLP(task=task, layer_arch=layer_arch, batch_size=batch_size,
                        activation_function=activation_function, activation_derivation=activation_derivation,
                        is_load=is_load, model_path=load_path)
    loss = RegressionBP.val(x_data, y_data)
    print("Test loss =", loss)