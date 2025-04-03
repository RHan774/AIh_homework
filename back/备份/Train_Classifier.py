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
random_range = config_train['init_params_random_range']
activation = config_train['activation_function']
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
    else:
        activation_function = my_utils.tanh
        activation_derivation = my_utils.tanh_derivation

    # 加载训练集和验证集的数据
    train_data = np.load(train_datapath)
    train_lables = np.load(train_labelpath)
    val_data = np.load(val_datapath)
    val_lables = np.load(val_labelpath)

    # 初始化模型
    ClassifierBP  = MLP(task, layer_arch, lr, batch_size, random_range,
                        activation_function, activation_derivation, is_load, save_path)
    
    max_accuracy = 0.979   # 当前最大的正确率
    # 训练epoches步
    for epoch in range(1, epoches + 1):
        # 每次训练batch_size大小的数据
        for batch_data, batch_labels in my_utils.get_batch(train_data, train_lables, batch_size, shuffle=True):
            loss = ClassifierBP.train(batch_data, batch_labels)
            
        # if epoch % 10 == 0:
        train_loss, train_accuracy = ClassifierBP.val(train_data, train_lables)
        val_loss, val_accuracy = ClassifierBP.val(val_data, val_lables)
        print("Epoch", epoch, "Train Accuracy =", train_accuracy, "loss =", train_loss)
        print("Epoch", epoch, "Val   Accuracy =", val_accuracy, "loss =", val_loss)
        if val_accuracy > max_accuracy:
            ClassifierBP.save_model()
            max_accuracy = val_accuracy
        