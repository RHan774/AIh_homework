import torch
import torch.nn as nn
import numpy as np
import os
from Model_torch import CNNTorch
import my_utils

def test(config_test):
    """测试模型"""
    # 测试参数
    batch_size = config_test['batch_size']

    # CNN架构参数
    in_channels = config_test['conv_channels'][0]  # 输入通道数是第一个元素
    conv_channels = config_test['conv_channels'][1:]  # 剩下的是卷积层通道数
    conv_kernel_sizes = config_test['conv_kernel_sizes']
    conv_strides = config_test['conv_strides']
    conv_paddings = config_test['conv_paddings']
    pool_sizes = config_test['pool_sizes']
    pool_strides = config_test['pool_strides']
    fc_sizes = config_test['fc_sizes']

    # 数据和模型路径
    test_datapath = config_test['data_path']
    test_labelpath = config_test['label_path']
    model_path = os.path.join(config_test['model_path'], 'model_torch.pth')
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 准备数据
    test_data, test_labels = my_utils.prepare_data(test_datapath, test_labelpath)
    
    # 创建数据加载器
    test_loader = my_utils.create_data_loaders(test_data, test_labels, batch_size, shuffle_train=False)
    
    # 获取设备
    device = my_utils.get_device()
    print(f"使用设备: {device}")
    
    # 初始化模型
    model = CNNTorch(
        in_channels=in_channels,
        conv_channels=conv_channels,
        conv_kernel_sizes=conv_kernel_sizes,
        conv_strides=conv_strides,
        conv_paddings=conv_paddings,
        pool_sizes=pool_sizes,
        pool_strides=pool_strides,
        fc_sizes=fc_sizes,
        use_dropout=False  # 测试时不使用dropout
    )
    
    # 加载模型
    model.load_model(model_path)
    
    # 移动模型到设备
    model.to(device)
    
    # 测试模型
    accuracy, all_preds, all_targets, _ = my_utils.test_model(model, test_loader, device)
    
    print(f'测试准确率: {accuracy:.2f}%')
    
    # 计算正确分类的样本数量
    correct = int(accuracy * len(test_labels) / 100)
    total = len(test_labels)
    
    # 绘制混淆矩阵
    confusion_matrix_path = os.path.join(os.path.dirname(model_path), 'confusion_matrix.png')
    my_utils.plot_confusion_matrix(all_targets, all_preds, confusion_matrix_path)
    
    # 保存测试结果
    result_path = os.path.join(os.path.dirname(model_path), 'test_results.txt')
    my_utils.save_test_results(accuracy, correct, total, result_path)
    
    # 可视化错误分类的样本
    error_samples_path = os.path.join(os.path.dirname(model_path), 'error_samples.png')
    my_utils.visualize_errors(test_data, all_targets, all_preds, error_samples_path)
    
    return accuracy

if __name__ == "__main__":
    dataset = 'MNIST'
    config = my_utils.read_config(dataset)
    config_test = config['Test']
    
    test(config_test) 