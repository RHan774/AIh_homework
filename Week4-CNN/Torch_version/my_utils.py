import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import yaml
from easydict import EasyDict
import argparse
import matplotlib.pyplot as plt

def read_config(task_kind):
    """
    读取配置文件
    
    参数:
        task_kind: 任务类型，比如'MNIST'或'CIFAR10'
        
    返回:
        配置字典
    """
    parser = argparse.ArgumentParser(description=task_kind)
    parser.add_argument("--config_path", type=str, default="config.yaml")
    args = parser.parse_args()
    config_path = args.config_path
    config = yaml.load(open(config_path, 'r', encoding='utf-8'), Loader=yaml.Loader)
    config = EasyDict(config)
    # 设置随机种子
    if 'Global' in config:
        np.random.seed(config.Global.random_seed)
        torch.manual_seed(config.Global.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.Global.random_seed)
    config = config[task_kind]
    return config

def prepare_data(data_path, label_path, is_rgb=False):
    """
    准备数据：加载并预处理数据
    
    参数:
        data_path: 数据文件路径
        label_path: 标签文件路径
        is_rgb: 是否为RGB图像 (CIFAR10等)
        
    返回:
        (data_tensor, label_tensor): 处理后的数据和标签张量
    """
    # 加载数据
    data = np.load(data_path)
    labels = np.load(label_path)
    
    # 如果是RGB图像且通道在最后一维，转换为PyTorch的NCHW格式
    if is_rgb and data.shape[-1] == 3:
        data = np.transpose(data, (0, 3, 1, 2))
    
    # 转换为PyTorch张量
    data_tensor = torch.tensor(data, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    
    # 转换标签格式（如果是one-hot编码）
    if label_tensor.dim() == 2:
        label_tensor = torch.argmax(label_tensor, dim=1)
    
    return data_tensor, label_tensor

def create_data_loaders(data, labels, batch_size, shuffle_train=True):
    """
    创建数据加载器
    
    参数:
        data: 数据张量
        labels: 标签张量
        batch_size: 批处理大小
        shuffle_train: 是否打乱训练数据
        
    返回:
        数据加载器
    """
    dataset = torch.utils.data.TensorDataset(data, labels)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle_train)
    return data_loader

def get_device():
    """
    获取可用的设备（CPU或GPU）
    
    返回:
        torch设备对象
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def initialize_model(model, init_method):
    """
    初始化模型权重
    
    参数:
        model: 要初始化的模型
        init_method: 初始化方法 (1=uniform, 2=normal, 3=he, 其他=xavier)
        
    返回:
        初始化后的模型
    """
    if init_method == 1:
        model._initialize_weights('uniform')
    elif init_method == 2:
        model._initialize_weights('normal')
    elif init_method == 3:
        model._initialize_weights('he')
    else:
        model._initialize_weights('xavier')
    return model

def create_optimizer(model, learning_rate, is_cifar=False):
    """
    创建优化器
    
    参数:
        model: 模型
        learning_rate: 学习率
        is_cifar: 是否是CIFAR10数据集（增加权重衰减）
        
    返回:
        优化器
    """
    if is_cifar:
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    else:
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

def create_scheduler(optimizer, epochs, learning_rate, min_lr_ratio, use_cosine_decay, is_cifar=False):
    """
    创建学习率调度器
    
    参数:
        optimizer: 优化器
        epochs: 总轮数
        learning_rate: 初始学习率
        min_lr_ratio: 最小学习率比例
        use_cosine_decay: 是否使用余弦衰减
        is_cifar: 是否是CIFAR10数据集
        
    返回:
        学习率调度器
    """
    if use_cosine_decay:
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=learning_rate * min_lr_ratio)
    else:
        step_size = 30 if is_cifar else 10
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

def ensure_correct_tensor_format(tensor, device):
    """
    确保张量格式正确（NCHW格式，并移动到指定设备）
    
    参数:
        tensor: 输入张量
        device: 目标设备
        
    返回:
        格式化后的张量
    """
    # 移动到设备
    tensor = tensor.to(device)
    
    # 如果输入是NHWC格式，转换为NCHW
    if tensor.dim() == 4 and tensor.shape[1] not in [1, 3]:
        tensor = tensor.permute(0, 3, 1, 2)
    
    return tensor

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    训练一个epoch
    
    参数:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        
    返回:
        (train_loss, train_acc): 训练损失和准确率
    """
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        # 确保数据格式正确并移动到设备
        inputs = ensure_correct_tensor_format(inputs, device)
        targets = targets.to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    # 计算训练损失和准确率
    train_loss = train_loss / len(train_loader)
    train_acc = 100. * correct / total
    
    return train_loss, train_acc

def train(model, config_train, config_val):
    """训练模型"""
    # 训练参数
    batch_size = config_train['batch_size']
    epochs = config_train['epoches']
    learning_rate = config_train['learning_rate']

    # 数据和模型路径
    train_datapath = config_train['data_path']
    train_labelpath = config_train['label_path']
    val_datapath = config_val['data_path']
    val_labelpath = config_val['label_path']
    save_path = os.path.join(config_train['model_path'], 'model_torch.pth')
    
    # 准备数据
    train_data, train_labels = prepare_data(train_datapath, train_labelpath)
    val_data, val_labels = prepare_data(val_datapath, val_labelpath)
    
    # 创建数据加载器
    train_loader = create_data_loaders(train_data, train_labels, batch_size, shuffle_train=True)
    val_loader = create_data_loaders(val_data, val_labels, batch_size, shuffle_train=False)
    
    # 获取设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 初始化权重
    initialize_model(model, config_train['init_method'])
    
    # 移动模型到设备
    model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, learning_rate, is_cifar=False)
    
    # 学习率调度
    use_lr_scheduler = config_train['use_lr_scheduler']
    if use_lr_scheduler:
        scheduler = create_scheduler(
            optimizer, epochs, learning_rate, config_train['min_lr_ratio'], 
            config_train['use_cosine_decay'], is_cifar=False)
    
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
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 保存训练历史
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 更新学习率
        if use_lr_scheduler:
            scheduler.step()
        
        # 打印进度
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            print(f'验证准确率提升 ({best_val_acc:.2f}% --> {val_acc:.2f}%)，保存模型...')
            model.save_model(save_path)
            best_val_acc = val_acc
    
    # 绘制训练曲线
    curves_path = os.path.join(os.path.dirname(save_path), 'training_curves.png')
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, curves_path)
    
    print(f"训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    print(f"模型保存于: {save_path}")


def validate(model, val_loader, criterion, device):
    """
    验证模型性能
    
    参数:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        
    返回:
        (val_loss, val_acc): 验证损失和准确率
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            # 确保数据格式正确并移动到设备
            inputs = ensure_correct_tensor_format(inputs, device)
            targets = targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 统计
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # 计算验证损失和准确率
    val_loss = val_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

def test_model(model, test_loader, device, class_names=None):
    """
    测试模型性能
    
    参数:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        class_names: 类别名称列表
        
    返回:
        (accuracy, all_preds, all_targets, class_accuracy): 准确率、所有预测、所有目标、每个类别的准确率
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    # 如果有类别名称，则记录每个类别的准确率
    if class_names:
        n_classes = len(class_names)
        class_correct = [0] * n_classes
        class_total = [0] * n_classes
    else:
        class_correct = None
        class_total = None
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            # 确保数据格式正确并移动到设备
            inputs = ensure_correct_tensor_format(inputs, device)
            targets = targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # 统计
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 记录每个类别的准确率
            if class_names:
                for i in range(targets.size(0)):
                    label = targets[i].item()
                    pred = predicted[i].item()
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1
            
            # 保存预测和真实标签
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # 计算准确率
    accuracy = 100. * correct / total
    
    # 计算每个类别的准确率
    class_accuracy = None
    if class_names:
        class_accuracy = [100 * class_correct[i] / max(1, class_total[i]) for i in range(n_classes)]
    
    return accuracy, all_preds, all_targets, class_accuracy

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """
    绘制训练曲线
    
    参数:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_accs: 训练准确率列表
        val_accs: 验证准确率列表
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    # 保存图表
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    print(f"训练曲线保存于: {save_path}")

def visualize_errors(data, true_labels, pred_labels, save_path, class_names=None, num_samples=5):
    """
    可视化错误分类的样本
    
    参数:
        data: 数据张量
        true_labels: 真实标签列表
        pred_labels: 预测标签列表
        save_path: 保存路径
        class_names: 类别名称列表
        num_samples: 样本数量
    """
    # 找到错误分类的样本索引
    error_indices = np.where(np.array(true_labels) != np.array(pred_labels))[0]
    
    if len(error_indices) == 0:
        print("没有错误分类的样本！")
        return
    
    # 限制样本数量
    num_samples = min(num_samples, len(error_indices))
    selected_indices = np.random.choice(error_indices, num_samples, replace=False)
    
    # 可视化错误分类的样本
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(selected_indices):
        plt.subplot(1, num_samples, i + 1)
        
        # 获取样本
        img = data[idx]
        
        # 处理不同的数据格式
        if isinstance(img, torch.Tensor):
            if img.shape[0] == 1 or img.shape[0] == 3:  # 如果是CHW格式
                img = img.permute(1, 2, 0)
            img = img.cpu().numpy()
        
        # 显示图像
        if img.shape[-1] == 1:  # 灰度图像
            plt.imshow(img.squeeze(), cmap='gray')
        else:  # 彩色图像
            plt.imshow(img)
        
        # 设置标题
        if class_names:
            plt.title(f'True: {class_names[true_labels[idx]]}\nPred: {class_names[pred_labels[idx]]}')
        else:
            plt.title(f'True: {true_labels[idx]}\nPred: {pred_labels[idx]}')
        
        plt.axis('off')
    
    # 保存图像
    plt.savefig(save_path)
    plt.close()
    
    print(f"错误分类样本可视化保存于: {save_path}")

def visualize_samples(data, labels, save_path, class_names=None, num_samples=10):
    """
    可视化样本图像
    
    参数:
        data: 数据张量
        labels: 标签列表
        save_path: 保存路径
        class_names: 类别名称列表
        num_samples: 样本数量
    """
    # 随机选择样本
    indices = np.random.choice(len(data), num_samples, replace=False)
    
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i + 1)
        
        # 获取样本
        img = data[idx]
        
        # 处理不同的数据格式
        if isinstance(img, torch.Tensor):
            if img.shape[0] == 1 or img.shape[0] == 3:  # 如果是CHW格式
                img = img.permute(1, 2, 0)
            img = img.cpu().numpy()
        
        # 显示图像
        if img.shape[-1] == 1:  # 灰度图像
            plt.imshow(img.squeeze(), cmap='gray')
        else:  # 彩色图像
            plt.imshow(img)
        
        # 设置标题
        if class_names:
            plt.title(class_names[labels[idx]])
        else:
            plt.title(f'Label: {labels[idx]}')
        
        plt.axis('off')
    
    # 保存图像
    plt.savefig(save_path)
    plt.close()
    
    print(f"样本图像保存于: {save_path}")

def save_test_results(accuracy, correct, total, save_path, class_accuracy=None, class_names=None):
    """
    保存测试结果
    
    参数:
        accuracy: 准确率
        correct: 正确分类的样本数
        total: 总样本数
        save_path: 保存路径
        class_accuracy: 每个类别的准确率列表
        class_names: 类别名称列表
    """
    with open(save_path, 'w') as f:
        f.write(f'测试准确率: {accuracy:.2f}%\n')
        f.write(f'正确分类样本数: {correct}/{total}\n\n')
        
        if class_accuracy and class_names:
            f.write("各类别的准确率:\n")
            for i, (name, acc) in enumerate(zip(class_names, class_accuracy)):
                f.write(f'{name}: {acc:.2f}%\n')
    
    print(f"测试结果保存于: {save_path}") 