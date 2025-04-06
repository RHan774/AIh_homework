import torch
import torch.nn as nn
import numpy as np
import os
import my_utils

class CNN(nn.Module):
    def __init__(self, batch_size, 
                 conv_channels, conv_kernel_sizes, conv_strides,
                 pool_sizes, pool_strides,
                 fc_sizes,
                 activation_function, activation_derivation,
                 conv_paddings=None, use_dropout=False, dropout_rates=None,
                 init_method=3, random_range=0.15,
                 lr=0.01, use_lr_scheduler=False, use_cosine_decay=False, 
                 use_warmup=False, warmup_epochs=0, min_lr_ratio=0.01, total_epochs=100,
                 is_load=False, model_path=''):
        """
        初始化CNN模型
        
        Args:
            batch_size: 批大小
            conv_channels: 卷积层通道列表
            conv_kernel_sizes: 卷积核大小列表
            conv_strides: 卷积步长列表
            pool_sizes: 池化核大小列表
            pool_strides: 池化步长列表
            fc_sizes: 全连接层大小列表，最后一个是输出类别数
            activation_function: 激活函数
            activation_derivation: 激活函数导数
            conv_paddings: 卷积层填充大小列表，若为None则默认为全0（无填充）
            use_dropout: 是否使用dropout
            dropout_rates: dropout概率列表，[池化层dropout, 全连接层dropout]
            init_method: 初始化方法
            random_range: 随机初始化范围
            lr: 学习率
            use_lr_scheduler: 是否使用学习率调度
            use_cosine_decay: 是否使用余弦衰减
            use_warmup: 是否使用预热
            warmup_epochs: 预热epoch数
            min_lr_ratio: 最小学习率比例
            total_epochs: 总训练epoch数
            is_load: 是否加载模型
            model_path: 模型路径
        """
        super(CNN, self).__init__()
        
        self.batch_size = batch_size
        self.init_lr = lr
        self.lr = lr
        self.use_dropout = use_dropout
        self.dropout_rates = dropout_rates if dropout_rates else [0.0, 0.0]
        
        # 处理卷积层填充参数，若未提供，则默认为0（无填充）
        if conv_paddings is None:
            conv_paddings = [0] * (len(conv_channels) - 1)
        
        # 学习率调度参数
        self.use_lr_scheduler = use_lr_scheduler
        self.use_cosine_decay = use_cosine_decay
        self.use_warmup = use_warmup
        self.warmup_epochs = warmup_epochs
        self.min_lr_ratio = min_lr_ratio
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
        # 模型加载参数
        self.is_load = is_load
        self.model_path = model_path
        
        # 激活函数
        self.activation_function = activation_function
        self.activation_derivation = activation_derivation
        
        # 根据激活函数选择PyTorch激活函数
        if isinstance(activation_function, type(my_utils.relu)):
            if activation_function == my_utils.relu:
                self.torch_activation = nn.ReLU()
            elif activation_function == my_utils.sigmoid:
                self.torch_activation = nn.Sigmoid()
            elif activation_function == my_utils.tanh:
                self.torch_activation = nn.Tanh()
            elif activation_function == my_utils.leaky_relu:
                self.torch_activation = nn.LeakyReLU(0.01)
            else:
                self.torch_activation = nn.ReLU()
        else:
            self.torch_activation = nn.ReLU()
            
        # 构建网络结构
        self.features = nn.Sequential()
        
        # 创建卷积和池化层
        conv_layer_count = len(conv_channels) - 1
        for i in range(conv_layer_count):
            # 添加卷积层
            self.features.add_module(
                f'conv{i}',
                nn.Conv2d(
                    in_channels=conv_channels[i],
                    out_channels=conv_channels[i+1],
                    kernel_size=conv_kernel_sizes[i],
                    stride=conv_strides[i],
                    padding=conv_paddings[i]
                )
            )
            
            # 添加激活函数
            self.features.add_module(f'act{i}', self.torch_activation)
            
            # 添加池化层
            self.features.add_module(
                f'pool{i}',
                nn.MaxPool2d(
                    kernel_size=pool_sizes[i],
                    stride=pool_strides[i]
                )
            )
            
            # 为池化层后添加dropout
            if use_dropout:
                self.features.add_module(
                    f'dropout{i}',
                    nn.Dropout(p=self.dropout_rates[0])
                )
        
        # 添加Flatten层
        self.features.add_module('flatten', nn.Flatten())
        
        # 添加全连接层和dropout
        self.fc = nn.Sequential()
        fc_layer_count = len(fc_sizes) - 1
        for i in range(fc_layer_count):
            # 确定该层输入大小和输出大小
            input_size = fc_sizes[i]
            output_size = fc_sizes[i+1]
            
            # 创建全连接层
            self.fc.add_module(
                f'fc{i}',
                nn.Linear(input_size, output_size)
            )
            
            # 如果不是最后一层，添加激活函数和dropout
            if i < fc_layer_count - 1:
                self.fc.add_module(f'act_fc{i}', self.torch_activation)
                
                # 添加dropout
                if use_dropout:
                    self.fc.add_module(
                        f'dropout_fc{i}',
                        nn.Dropout(p=self.dropout_rates[1])
                    )
        
        # 初始化权重
        self._initialize_weights(init_method, random_range)
        
        # 如果需要加载模型
        if self.is_load:
            self.load_model()
        
        # 创建优化器
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
            
    def _initialize_weights(self, init_method, random_range):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if init_method == 0:  # 零初始化
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif init_method == 1:  # 均匀随机初始化
                    nn.init.uniform_(m.weight, -random_range, random_range)
                    if m.bias is not None:
                        nn.init.uniform_(m.bias, -random_range, random_range)
                        nn.init.constant_(m.bias, -1)  # 将bias先置为负数
                elif init_method == 2:  # 正态分布初始化
                    nn.init.normal_(m.weight, 0, random_range)
                    if m.bias is not None:
                        nn.init.normal_(m.bias, 0, random_range)
                        nn.init.constant_(m.bias, -1)  # 将bias先置为负数
                elif init_method == 3:  # He初始化
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:  # Xavier/Glorot初始化 (默认)
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def _set_training_mode(self, is_training):
        """设置网络训练模式"""
        if is_training:
            self.train()
        else:
            self.eval()
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入数据，形状为 (batch_size, channels, height, width)
            
        Returns:
            模型输出
        """
        # 处理输入数据类型
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            
        # 确保数据的维度顺序正确 (Numpy: NHWC -> PyTorch: NCHW)
        if x.dim() == 4 and x.shape[1] not in [1, 3]:  # 可能是NHWC格式
            x = x.permute(0, 3, 1, 2)
            
        # 通过特征提取层
        x = self.features(x)
        
        # 通过分类器层
        x = self.fc(x)
        
        return x
    
    def train(self, batch_data, batch_labels):
        """
        训练一个batch
        
        Args:
            batch_data: 一个batch的输入数据
            batch_labels: 一个batch的标签
            
        Returns:
            损失值
        """
        self._set_training_mode(True)
        
        # 转换为PyTorch张量
        inputs = torch.tensor(batch_data, dtype=torch.float32)
        labels = torch.tensor(batch_labels, dtype=torch.long)
        
        # 数据格式转换(如果需要)
        if inputs.dim() == 4 and inputs.shape[1] not in [1, 3]:  # 如果是NHWC格式
            inputs = inputs.permute(0, 3, 1, 2)  # 转换为NCHW格式
        
        # 清除梯度
        self.optimizer.zero_grad()
        
        # 前向传播
        outputs = self(inputs)
        
        # 计算损失
        criterion = nn.CrossEntropyLoss()
        if labels.dim() == 2:  # 如果是one-hot编码
            labels = torch.argmax(labels, dim=1)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, train_data, train_labels):
        """
        训练一个完整的epoch
        
        Args:
            train_data: 训练数据
            train_labels: 训练标签
            
        Returns:
            无
        """
        self._set_training_mode(True)
        
        # 更新学习率
        if self.use_lr_scheduler:
            self.lr = my_utils.get_lr_by_scheduler(
                self.init_lr,
                self.current_epoch,
                self.total_epochs,
                self.warmup_epochs,
                self.use_warmup,
                self.use_cosine_decay,
                self.min_lr_ratio
            )
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
        
        # 获取batch数据
        batches = my_utils.get_batch(train_data, train_labels, self.batch_size, shuffle=True)
        
        # 训练每个batch
        losses = []
        for batch_data, batch_labels in batches:
            loss = self.train(batch_data, batch_labels)
            losses.append(loss)
        
        # 更新epoch计数
        self.current_epoch += 1
        
        return np.mean(losses)
    
    def val(self, val_data, val_labels):
        """
        在验证集上评估模型
        
        Args:
            val_data: 验证数据
            val_labels: 验证标签
            
        Returns:
            (loss, accuracy): 验证损失和准确率
        """
        self._set_training_mode(False)
        
        # 转换为PyTorch张量
        inputs = torch.tensor(val_data, dtype=torch.float32)
        labels = torch.tensor(val_labels, dtype=torch.long)
        
        # 数据格式转换(如果需要)
        if inputs.dim() == 4 and inputs.shape[1] not in [1, 3]:  # 如果是NHWC格式
            inputs = inputs.permute(0, 3, 1, 2)  # 转换为NCHW格式
        
        with torch.no_grad():
            # 前向传播
            outputs = self(inputs)
            
            # 计算损失
            criterion = nn.CrossEntropyLoss()
            if labels.dim() == 2:  # 如果是one-hot编码
                targets = torch.argmax(labels, dim=1)
            else:
                targets = labels
            loss = criterion(outputs, targets)
            
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == targets).sum().item()
            accuracy = correct / labels.shape[0]
            
        return loss.item(), accuracy
    
    def save_model(self):
        """保存模型参数"""
        # 创建模型目录
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
        # 保存模型参数
        torch.save(self.state_dict(), os.path.join(self.model_path, 'model.pth'))
    
    def load_model(self):
        """加载模型参数"""
        model_file = os.path.join(self.model_path, 'model.pth')
        if os.path.exists(model_file):
            self.load_state_dict(torch.load(model_file))
        else:
            print(f"警告: 模型文件 {model_file} 不存在，使用随机初始化。")


##### Model Evaluation #####
def evaluate(imgs, labels, model):
    """
    评估模型性能
    
    Args:
        imgs: 输入图像
        labels: 真实标签
        model: 模型
        
    Returns:
        pred_label: 预测标签
    """
    # 设置为评估模式
    model._set_training_mode(False)
    
    # 转换为PyTorch张量
    inputs = torch.tensor(imgs, dtype=torch.float32)
    
    # 数据格式转换(如果需要)
    if inputs.dim() == 4 and inputs.shape[1] not in [1, 3]:  # 如果是NHWC格式
        inputs = inputs.permute(0, 3, 1, 2)  # 转换为NCHW格式
    
    with torch.no_grad():
        # 获取模型输出
        outputs = model(inputs)
        
        # 获取预测标签
        _, pred_label = torch.max(outputs, 1)
        pred_label = pred_label.numpy()
        
        # 计算准确率
        if isinstance(labels, torch.Tensor):
            true_labels = labels
        else:
            true_labels = torch.tensor(labels)
            
        if true_labels.dim() == 2:  # 如果是one-hot编码
            true_labels = torch.argmax(true_labels, dim=1)
            
        correct_cnt = (pred_label == true_labels.numpy()).sum()
        print(f'match rate: {correct_cnt/labels.shape[0]}')
        
    return pred_label
