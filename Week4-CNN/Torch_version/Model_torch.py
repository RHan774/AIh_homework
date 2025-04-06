import torch
import torch.nn as nn
import os

class CNNTorch(nn.Module):
    def __init__(self, conv_channels, conv_kernel_sizes, conv_strides,
                 conv_paddings, pool_sizes, pool_strides, fc_sizes, 
                 dropout_rate=0.0, use_dropout=False):
        """
        初始化CNN模型

        参数:
            conv_channels: 卷积层通道数列表 (第一个是输入通道数)
            conv_kernel_sizes: 卷积核大小列表
            conv_strides: 卷积步长列表
            conv_paddings: 卷积填充大小列表
            pool_sizes: 池化核大小列表
            pool_strides: 池化步长列表
            fc_sizes: 全连接层大小列表 (第一个是最后的池化层输出大小，最后一个是类别数)
            dropout_rate: dropout比率
            use_dropout: 是否使用dropout
        """
        super(CNNTorch, self).__init__()
        
        # 保存配置参数
        self.conv_channels = conv_channels
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_strides = conv_strides
        self.conv_paddings = conv_paddings
        self.pool_sizes = pool_sizes
        self.pool_strides = pool_strides
        self.fc_sizes = fc_sizes
        self.dropout_rate = dropout_rate
        self.use_dropout = use_dropout
        
        # 构建卷积特征提取层
        layers = []
        
        # 添加所有卷积层
        for i in range(len(conv_channels) - 1):
            # 添加卷积层
            layers.append(
                nn.Conv2d(
                    in_channels=conv_channels[i],
                    out_channels=conv_channels[i+1],
                    kernel_size=conv_kernel_sizes[i],
                    stride=conv_strides[i],
                    padding=conv_paddings[i]
                )
            )
            
            # 添加ReLU激活函数
            layers.append(nn.ReLU())
            
            # 添加池化层
            layers.append(
                nn.MaxPool2d(
                    kernel_size=pool_sizes[i],
                    stride=pool_strides[i]
                )
            )
            
            # 如果需要使用dropout
            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))
            
            
        # 添加Flatten层
        layers.append(nn.Flatten())
        
        # 构建特征提取层
        self.features = nn.Sequential(*layers)
        
        # 构建全连接层
        fc_layers = []
        
        # 添加全连接层（使用fc_sizes中提供的大小）
        for i in range(len(fc_sizes) - 1):
            # 添加全连接层
            fc_layers.append(nn.Linear(fc_sizes[i], fc_sizes[i+1]))
            
            # 除了最后一层，都添加ReLU激活函数和可选的dropout
            if i < len(fc_sizes) - 2:
                fc_layers.append(nn.ReLU())
                
                # 如果需要使用dropout
                if use_dropout:
                    fc_layers.append(nn.Dropout(dropout_rate))
        
        # 构建全连接层
        self.fc = nn.Sequential(*fc_layers)
    
    def forward(self, x):
        """前向传播"""
        # 提取特征
        features = self.features(x)
        
        # 通过全连接层
        x = self.fc(features)
        
        return x
    
    def _initialize_weights(self, method='he'):
        """初始化网络权重
        
        参数:
            method: 'he', 'xavier', 'normal', 'uniform'中的一种
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if method == 'he':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif method == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                elif method == 'normal':
                    nn.init.normal_(m.weight, 0, 0.01)
                elif method == 'uniform':
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def save_model(self, path):
        """保存模型
        
        参数:
            path: 保存路径
        """
        # 创建目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        """加载模型
        
        参数:
            path: 模型路径
        """
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
        else:
            print(f"警告: 模型文件 {path} 不存在") 