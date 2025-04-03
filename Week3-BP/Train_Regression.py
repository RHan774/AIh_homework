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
init_method = config_train['init_method']
random_range = config_train['init_params_random_range']
activation = config_train['activation_function']
use_dropout = config_train['use_dropout']
dropout_rates = config_train['dropout_rates']

# 学习率调度相关配置
use_lr_scheduler = config_train['use_lr_scheduler']
use_cosine_decay = config_train['use_cosine_decay']
use_warmup = config_train['use_warmup']
warmup_epochs = config_train['warmup_epochs']
min_lr_ratio = config_train['min_lr_ratio']

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
    elif activation == 2:
        activation_function = my_utils.tanh
        activation_derivation = my_utils.tanh_derivation
    elif activation == 3:
        activation_function = my_utils.leaky_relu
        activation_derivation = my_utils.leaky_relu_derivation
    elif activation == 4:
        activation_function = my_utils.prelu
        activation_derivation = my_utils.prelu_derivation
    elif activation == 5:
        activation_function = my_utils.gelu
        activation_derivation = my_utils.gelu_derivation
    elif activation == 6:
        activation_function = my_utils.swish
        activation_derivation = my_utils.swish_derivation
    elif activation == 7:
        activation_function = my_utils.swiglu
        activation_derivation = my_utils.swiglu_derivation

    # 加载训练集和验证集的数据
    train_x = np.load(train_xpath)
    train_y = np.load(train_ypath)
    val_x = np.load(val_datapath)
    val_y = np.load(val_labelpath)

    # 初始化模型
    RegressionBP = MLP(task, layer_arch, batch_size, lr, init_method, random_range,
                      activation_function, activation_derivation, 
                      use_dropout, dropout_rates, 
                      use_lr_scheduler, use_cosine_decay, use_warmup,
                      warmup_epochs, min_lr_ratio, epoches,
                      is_load, save_path)
    
    min_loss = 5.0683559947185754e-06   # 当前最小的均方误差
    # 训练epoches步
    for epoch in range(1, epoches + 1):
        # 训练一个epoch
        RegressionBP.train_epoch(train_x, train_y)
            
        if epoch % 10 == 0:
            train_loss = RegressionBP.val(train_x, train_y)
            val_loss = RegressionBP.val(val_x, val_y)
            print("Epoch", epoch, "Train loss =", train_loss)
            print("Epoch", epoch, "Val   loss =", val_loss)
            if val_loss < min_loss:
                RegressionBP.save_model()
                min_loss = val_loss
            