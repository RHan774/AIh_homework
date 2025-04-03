import my_utils
import numpy as np
from Modle import MLP

# 从config文件中读取配置的参数
task = 'Classifier'
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
data_path = config_test['data_path']
label_path = config_test['label_path']

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
    data = np.load(data_path)
    lable = np.load(label_path)

    # 初始化模型
    ClassifierBP  = MLP(task, layer_arch, lr, batch_size, random_range,
                        activation_function, activation_derivation, is_load, load_path)

    loss, accuracy = ClassifierBP.val(data, lable)
    print("Test Accuracy =", accuracy, "loss =", loss)