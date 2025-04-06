import torch
import torch.nn as nn
import numpy as np
import os
from Model_torch import CNNTorch
import my_utils

# 设置环境变量，解决OpenMP警告
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def train(config_train, config_val):
    """训练模型"""
    # 训练参数
    batch_size = config_train['batch_size']
    epochs = config_train['epoches']
    learning_rate = config_train['learning_rate']
    use_dropout = config_train['use_dropout']
    dropout_rates = config_train['dropout_rates'][0] if len(config_train['dropout_rates']) > 0 else 0.0

    # CNN架构参数
    in_channels = config_train['conv_channels'][0]  # 输入通道数是第一个元素
    conv_channels = config_train['conv_channels'][1:]  # 剩下的是卷积层通道数
    conv_kernel_sizes = config_train['conv_kernel_sizes']
    conv_strides = config_train['conv_strides']
    conv_paddings = config_train['conv_paddings']
    pool_sizes = config_train['pool_sizes']
    pool_strides = config_train['pool_strides']
    fc_sizes = config_train['fc_sizes']

    # 数据和模型路径
    train_datapath = config_train['data_path']
    train_labelpath = config_train['label_path']
    val_datapath = config_val['data_path']
    val_labelpath = config_val['label_path']
    save_path = os.path.join(config_train['model_path'], 'model_torch.pth')
    
    # 类别名称
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 准备数据 (CIFAR10是RGB图像)
    train_data, train_labels = my_utils.prepare_data(train_datapath, train_labelpath, is_rgb=True)
    val_data, val_labels = my_utils.prepare_data(val_datapath, val_labelpath, is_rgb=True)
    
    # 创建数据加载器
    train_loader = my_utils.create_data_loaders(train_data, train_labels, batch_size, shuffle_train=True)
    val_loader = my_utils.create_data_loaders(val_data, val_labels, batch_size, shuffle_train=False)
    
    # 获取设备
    device = my_utils.get_device()
    print(f"使用设备: {device}")
    
    # 初始化CNN模型
    model = CNNTorch(
        conv_channels=conv_channels,
        conv_kernel_sizes=conv_kernel_sizes,
        conv_strides=conv_strides,
        conv_paddings=conv_paddings,
        pool_sizes=pool_sizes,
        pool_strides=pool_strides,
        fc_sizes=fc_sizes,
        dropout_rate=dropout_rates,
        use_dropout=use_dropout
    )
    
    # 初始化权重
    my_utils.initialize_model(model, config_train['init_method'])
    
    # 移动模型到设备
    model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = my_utils.create_optimizer(model, learning_rate, is_cifar=True)
    
    # 学习率调度
    use_lr_scheduler = config_train['use_lr_scheduler']
    if use_lr_scheduler:
        scheduler = my_utils.create_scheduler(
            optimizer, epochs, learning_rate, config_train['min_lr_ratio'], 
            config_train['use_cosine_decay'], is_cifar=True)
    
    # 用于保存训练历史的列表
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # 记录最佳验证准确率
    best_val_acc = 0.0
    
    # 训练循环
    for epoch in range(epochs):
        # 训练一个epoch
        train_loss, train_acc = my_utils.train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc = my_utils.validate(model, val_loader, criterion, device)
        
        # 保存训练历史
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 更新学习率
        if use_lr_scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"当前学习率: {current_lr:.6f}")
        
        # 打印进度
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            print(f'验证准确率提升 ({best_val_acc:.2f}% --> {val_acc:.2f}%)，保存模型...')
            model.save_model(save_path)
            best_val_acc = val_acc
    
    # 绘制训练曲线
    curves_path = os.path.join(os.path.dirname(save_path), 'training_curves.png')
    my_utils.plot_training_curves(train_losses, val_losses, train_accs, val_accs, curves_path)
    
    # 可视化CIFAR10样本
    samples_path = os.path.join(os.path.dirname(save_path), 'cifar10_samples.png')
    my_utils.visualize_samples(train_data[:100], train_labels[:100], samples_path, class_names)
    
    print(f"训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    print(f"模型保存于: {save_path}")

if __name__ == "__main__":
    dataset = 'CIFAR10'
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
    
    # 调用训练函数，指定是CIFAR10数据集
    my_utils.train(model, config_train, config_val) 