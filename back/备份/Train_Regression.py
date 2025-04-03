import my_utils
import numpy as np
from Modle import MLP

# 从config文件中读取配置的参数
task = 'Regression'
config = my_utils.read_config(task)
config_train = config['Train']
config_val = config['Val']

batch_size = config_train['batch_size']
epoches = config_train['epoches']
layer_arch = config_train['layer_arch']
lr = config_train['learning_rate']
random_range = config_train['init_params_random_range']
activation = config_train['activation_function']
train_xpath = config_train['x_data']
train_ypath = config_train['y_data']
is_load = config_train['is_load']
save_path = config_train['model_path']
val_datapath = config_val['x_data']
val_labelpath = config_val['y_data']


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
    train_x = np.load(train_xpath)
    train_y = np.load(train_ypath)
    val_x = np.load(val_datapath)
    val_y = np.load(val_labelpath)

    # 初始化模型
    ClassifierBP  = MLP(task, layer_arch, lr, batch_size, random_range,
                        activation_function, activation_derivation, is_load, save_path)
    
    min_loss = 5.0683559947185754e-06   # 当前最小的均方误差
    # 训练epoches步
    for epoch in range(1, epoches + 1):
        # 每次训练batch_size大小的数据
        for batch_data, batch_labels in my_utils.get_batch(train_x, train_y, batch_size, shuffle=True):
            loss = ClassifierBP.train(batch_data, batch_labels)
            
        if epoch % 10 == 0:
            train_loss = ClassifierBP.val(train_x, train_y)
            val_loss = ClassifierBP.val(val_x, val_y)
            print("Epoch", epoch, "Train loss =", train_loss)
            print("Epoch", epoch, "Val   loss =", val_loss)
            if val_loss < min_loss:
                ClassifierBP.save_model()
                min_loss = val_loss
            