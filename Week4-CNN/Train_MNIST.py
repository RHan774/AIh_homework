import my_utils
import numpy as np
from Model import CNN

# 从config文件中读取配置的参数
dataset = 'MNIST'
config = my_utils.read_config(dataset)
config_train = config['Train']
config_val = config['Val']

# 训练参数
batch_size = config_train['batch_size']
epoches = config_train['epoches']
lr = config_train['learning_rate']
init_method = config_train['init_method']
random_range = config_train['init_params_random_range']
activation = config_train['activation_function']
use_dropout = config_train['use_dropout']
dropout_rates = config_train['dropout_rates']

# CNN架构参数
conv_channels = config_train['conv_channels']
conv_kernel_sizes = config_train['conv_kernel_sizes']
conv_strides = config_train['conv_strides']
conv_paddings = config_train['conv_paddings']
pool_sizes = config_train['pool_sizes']
pool_strides = config_train['pool_strides']
fc_sizes = config_train['fc_sizes']

# 学习率调度相关配置
use_lr_scheduler = config_train['use_lr_scheduler']
use_cosine_decay = config_train['use_cosine_decay']
use_warmup = config_train['use_warmup']
warmup_epochs = config_train['warmup_epochs']
min_lr_ratio = config_train['min_lr_ratio']

# 数据和模型路径
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

    # 加载训练集和验证集的数据
    train_data = np.load(train_datapath)
    train_labels = np.load(train_labelpath)
    val_data = np.load(val_datapath)
    val_labels = np.load(val_labelpath)
    # print(f"训练集: {train_data.shape}, {train_labels.shape}")
    # print(f"验证集: {val_data.shape}, {val_labels.shape}")

    # 初始化CNN模型
    mnist_cnn = CNN(
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
        use_dropout=use_dropout,
        dropout_rates=dropout_rates,
        init_method=init_method,
        random_range=random_range,
        lr=lr,
        use_lr_scheduler=use_lr_scheduler,
        use_cosine_decay=use_cosine_decay,
        use_warmup=use_warmup,
        warmup_epochs=warmup_epochs,
        min_lr_ratio=min_lr_ratio,
        total_epochs=epoches,
        is_load=is_load,
        model_path=save_path
    )
    
    max_accuracy = 0.0   # 当前最大的正确率
    # 训练epoches步
    for epoch in range(1, epoches + 1):
        # 训练一个epoch
        mnist_cnn.train_epoch(train_data, train_labels)
        
        if epoch % 5 == 0 or epoch == 1:
            train_loss, train_accuracy = mnist_cnn.val(train_data, train_labels)
            val_loss, val_accuracy = mnist_cnn.val(val_data, val_labels)
            print(f"Epoch {epoch}, 学习率: {mnist_cnn.lr}")
            print(f"训练集: 准确率 = {train_accuracy}, 损失 = {train_loss}")
            print(f"验证集: 准确率 = {val_accuracy}, 损失 = {val_loss}")
            
            # 保存最佳模型
            if val_accuracy > max_accuracy:
                mnist_cnn.save_model()
                max_accuracy = val_accuracy
                print(f"保存新的最佳模型，验证集准确率: {val_accuracy}")
            
            print("-" * 50) 