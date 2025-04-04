import my_utils
import numpy as np
from Modle import MLP

# 从config文件中读取配置的参数
task = 'Classifier'
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

train_datapath = config_train['data_path']
train_labelpath = config_train['label_path']
is_load = config_train['is_load']
save_path = config_train['model_path']
val_datapath = config_val['data_path']
val_labelpath = config_val['label_path']


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
    train_data = np.load(train_datapath)
    train_labels = np.load(train_labelpath)
    val_data = np.load(val_datapath)
    val_labels = np.load(val_labelpath)

    # 初始化模型
    ClassifierBP = MLP(task, layer_arch, batch_size, lr, init_method, random_range,
                      activation_function, activation_derivation, 
                      use_dropout, dropout_rates, 
                      use_lr_scheduler, use_cosine_decay, use_warmup,
                      warmup_epochs, min_lr_ratio, epoches,
                      is_load, save_path)
    
    max_accuracy = 0.9741   # 当前最大的正确率
    # 训练epoches步
    for epoch in range(1, epoches + 1):
        # 每次训练batch_size大小的数据
        for batch_data, batch_labels in my_utils.get_batch(train_data, train_labels, batch_size, shuffle=True):
            ClassifierBP.train(batch_data, batch_labels)

        # 应用学习率调度时用这个版本
        # ClassifierBP.train_epoch(train_data, train_labels)
        
        if epoch % 10 == 0:
            train_loss, train_accuracy = ClassifierBP.val(train_data, train_labels)
            val_loss, val_accuracy = ClassifierBP.val(val_data, val_labels)
            print("Epoch", epoch, "Train Accuracy =", train_accuracy, "loss =", train_loss)
            print("Epoch", epoch, "Val   Accuracy =", val_accuracy, "loss =", val_loss)
            if val_accuracy > max_accuracy:
                ClassifierBP.save_model()
                max_accuracy = val_accuracy
        