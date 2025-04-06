import numpy as np
import os
import my_utils

class CNN:
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
        
        # 构建网络结构
        self.conv_layers = []
        self.pool_layers = []
        self.fc_layers = []
        self.dropout_layers = []
        
        # 创建卷积和池化层
        conv_layer_count = len(conv_channels) - 1
        for i in range(conv_layer_count):
            # 添加卷积层
            self.conv_layers.append(
                Conv2D(
                    in_channels=conv_channels[i],
                    out_channels=conv_channels[i+1],
                    kernel_size=conv_kernel_sizes[i],
                    stride=conv_strides[i],
                    padding=conv_paddings[i],
                    init_method=init_method,
                    random_range=random_range
                )
            )
            
            # 添加池化层
            self.pool_layers.append(
                MaxPool2D(
                    pool_size=pool_sizes[i],
                    stride=pool_strides[i]
                )
            )
            
            # 为池化层添加dropout
            if use_dropout:
                self.dropout_layers.append(
                    Dropout(dropout_rate=self.dropout_rates[0])
                )
        
        # 添加Flatten层
        self.flatten_layer = Flatten()
        
        # 添加全连接层和dropout
        fc_layer_count = len(fc_sizes) - 1
        for i in range(fc_layer_count):
            # 确定该层输入大小和输出大小
            input_size = fc_sizes[i]
            output_size = fc_sizes[i+1]
            
            # 确定是否为输出层
            is_output = (i == fc_layer_count - 1)
            
            # 创建全连接层
            self.fc_layers.append(
                FullyConnected(
                    input_size=input_size,
                    output_size=output_size,
                    init_method=init_method,
                    random_range=random_range,
                    output_layer=is_output
                )
            )
            
            # 为非输出层的全连接层添加dropout
            if use_dropout and not is_output:
                self.dropout_layers.append(
                    Dropout(dropout_rate=self.dropout_rates[1])
                )
        
        # 如果需要加载模型
        if self.is_load:
            self.load_model()
    
    def _set_training_mode(self, is_training):
        """设置所有层的训练模式"""
        for layer in self.conv_layers:
            layer.set_training(is_training)
        
        for layer in self.pool_layers:
            layer.set_training(is_training)
        
        for layer in self.fc_layers:
            if layer is not None:
                layer.set_training(is_training)
        
        self.flatten_layer.set_training(is_training)
        
        for layer in self.dropout_layers:
            layer.set_training(is_training)
    
    def forward(self, input_data):
        """
        前向传播
        
        Args:
            input_data: 输入数据，形状为 (batch_size, height, width, channels)
            
        Returns:
            模型输出
        """
        # # 输入维度变换：如果输入是灰度图像，添加通道维度 （已经在data_process.py中转换了）
        # if len(input_data.shape) == 3:
        #     # 输入是(batch_size, height, width)
        #     input_data = input_data.reshape(input_data.shape[0], input_data.shape[1], input_data.shape[2], 1)
        
        # 前向传播通过卷积和池化层
        x = input_data
        dropout_idx = 0
        
        for i in range(len(self.conv_layers)):
            # 卷积层
            x = self.conv_layers[i].forward(x)
            # 激活函数
            x = self.activation_function(x)
            # 池化层
            x = self.pool_layers[i].forward(x)
            # Dropout (如果使用)
            if self.use_dropout:
                x = self.dropout_layers[dropout_idx].forward(x)
                dropout_idx += 1
        
        # 展平特征图
        x = self.flatten_layer.forward(x)
        
        # 前向传播通过全连接层
        for i in range(len(self.fc_layers)):
            # 全连接层
            x = self.fc_layers[i].forward(x, self.activation_function)
            # Dropout (如果使用且不是输出层)
            if self.use_dropout and i < len(self.fc_layers) - 1:
                x = self.dropout_layers[dropout_idx].forward(x)
                dropout_idx += 1
        
        return x
    
    def backward(self, loss):
        """
        反向传播
        
        Args:
            loss: 损失梯度
            
        Returns:
            无
        """
        # 从最后一层开始反向传播
        grad = loss
        
        # 通过全连接层反向传播
        for i in range(len(self.fc_layers) - 1, -1, -1):
            # 如果不是输出层且使用了dropout，先通过dropout层
            if self.use_dropout and i < len(self.fc_layers) - 1:
                dropout_idx = len(self.conv_layers) + i if self.use_dropout else 0
                grad = self.dropout_layers[dropout_idx].backward(grad)
            
            # 通过全连接层
            grad = self.fc_layers[i].backward(grad, self.activation_derivation)
        
        # 通过Flatten层
        grad = self.flatten_layer.backward(grad)
        
        # 通过卷积层和池化层
        for i in range(len(self.conv_layers) - 1, -1, -1):
            # 如果使用了dropout
            if self.use_dropout:
                dropout_idx = i
                grad = self.dropout_layers[dropout_idx].backward(grad)
            
            # 通过池化层
            grad = self.pool_layers[i].backward(grad)
            # 通过卷积层（包括激活函数导数的计算）
            grad = self.conv_layers[i].backward(grad)
    
    def update(self):
        """更新所有层的参数"""
        # 更新卷积层
        for layer in self.conv_layers:
            layer.update(self.lr)
        
        # 更新全连接层
        for layer in self.fc_layers:
            if layer is not None:
                layer.update(self.lr)
    
    def train(self, batch_data, batch_labels):
        """
        训练一个batch的数据
        
        Args:
            batch_data: 批次数据
            batch_labels: 批次标签
            
        Returns:
            损失和准确率
        """
        # 设置为训练模式
        self._set_training_mode(True)
        
        # 前向传播
        predictions = self.forward(batch_data)
        
        # 计算损失
        loss = batch_labels - predictions
        
        # 反向传播
        self.backward(loss)
        
        # 更新参数
        self.update()

    
    def train_epoch(self, train_data, train_labels):
        """
        训练一个epoch的数据
        
        Args:
            train_data: 训练数据
            train_labels: 训练标签
            
        Returns:
            无
        """
        # 如果使用学习率调度，更新学习率
        if self.use_lr_scheduler:
            self.lr = my_utils.get_lr_by_scheduler(
                self.init_lr, self.current_epoch, self.total_epochs,
                self.warmup_epochs, self.use_warmup,
                self.use_cosine_decay, self.min_lr_ratio
            )
        
        # 训练
        for batch_data, batch_labels in my_utils.get_batch(train_data, train_labels, self.batch_size, shuffle=True):
            self.train(batch_data, batch_labels)
        
        # 更新epoch计数
        self.current_epoch += 1
        

    
    def val(self, val_data, val_labels):
        """
        验证模型
        
        Args:
            val_data: 验证数据
            val_labels: 验证标签
            
        Returns:
            损失和准确率
        """
        # 设置为评估模式
        self._set_training_mode(False)
        
        # 存储所有批次的损失和准确率
        correct = 0
        loss_sum = 0
        total = len(val_labels)
        
        # 按批次处理数据
        for batch_data, batch_labels in my_utils.get_batch(val_data, val_labels, self.batch_size, shuffle=False):
            # 前向传播
            predictions = self.forward(batch_data)
            
            # 计算交叉熵损失 (防止log(0)，添加一个小的epsilon值)
            epsilon = 1e-10
            loss_sum += -np.sum(batch_labels * np.log(predictions + epsilon))
            
            # 计算准确率
            predictions_class = np.argmax(predictions, axis=1)
            labels_class = np.argmax(batch_labels, axis=1)
            correct += np.sum(predictions_class == labels_class)
        
        # 计算平均损失和准确率
        avg_loss = loss_sum / total
        avg_accuracy = correct / total
        
        return avg_loss, avg_accuracy
    
    def save_model(self):
        """保存模型"""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
        # 保存卷积层参数
        for i, layer in enumerate(self.conv_layers):
            np.save(f"{self.model_path}/conv_{i}_weights.npy", layer.get_weights())
            np.save(f"{self.model_path}/conv_{i}_bias.npy", layer.get_bias())
        
        # 保存全连接层参数
        for i, layer in enumerate(self.fc_layers):
            if layer is not None:
                np.save(f"{self.model_path}/fc_{i}_weights.npy", layer.get_weights())
                np.save(f"{self.model_path}/fc_{i}_bias.npy", layer.get_bias())
    
    def load_model(self):
        """加载模型"""
        # 加载卷积层参数
        for i, layer in enumerate(self.conv_layers):
            weight_path = f"{self.model_path}/conv_{i}_weights.npy"
            bias_path = f"{self.model_path}/conv_{i}_bias.npy"
            
            if os.path.exists(weight_path) and os.path.exists(bias_path):
                layer.set_weights(np.load(weight_path))
                layer.set_bias(np.load(bias_path))
        
        # 加载全连接层参数
        for i, layer in enumerate(self.fc_layers):
            if layer is not None:
                weight_path = f"{self.model_path}/fc_{i}_weights.npy"
                bias_path = f"{self.model_path}/fc_{i}_bias.npy"
                
                if os.path.exists(weight_path) and os.path.exists(bias_path):
                    layer.set_weights(np.load(weight_path))
                    layer.set_bias(np.load(bias_path))

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, init_method=3, random_range=0.15):
        """
        初始化卷积层
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充大小，0表示无填充
            init_method: 初始化方法
            random_range: 随机初始化的范围或标准差
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 初始化卷积核权重和偏置
        self._init_weights(init_method, random_range)
        
        # 存储中间结果用于反向传播
        self.input_data = None
        self.input_shape = None
        self.output_shape = None
        self.padded_input = None
        self.x_cols = None
        
        # 存储梯度
        self.delta_weights = np.zeros_like(self.weights)
        self.delta_bias = np.zeros_like(self.bias)
        
        # 训练模式标志
        self.is_training = True
    
    def _init_weights(self, init_method, random_range):
        """初始化卷积核权重"""
        if init_method == 0:
            # 零初始化
            self.weights = np.zeros((self.kernel_size, self.kernel_size, self.in_channels, self.out_channels))
            self.bias = np.zeros((1, 1, 1, self.out_channels))
        elif init_method == 1:
            # 均匀随机初始化
            self.weights = np.random.uniform(-random_range, random_range, 
                                           (self.kernel_size, self.kernel_size, self.in_channels, self.out_channels))
            self.bias = np.random.uniform(-random_range, random_range, (1, 1, 1, self.out_channels))
        elif init_method == 2:
            # 正态分布初始化
            self.weights = np.random.normal(0, random_range, 
                                          (self.kernel_size, self.kernel_size, self.in_channels, self.out_channels))
            self.bias = np.random.normal(0, random_range, (1, 1, 1, self.out_channels))
        else:
            # 何凯明初始化 (适用于ReLU激活函数)
            fan_in = self.kernel_size * self.kernel_size * self.in_channels
            he_std = np.sqrt(2.0 / fan_in)
            self.weights = np.random.normal(0, he_std, 
                                          (self.kernel_size, self.kernel_size, self.in_channels, self.out_channels))
            self.bias = np.random.normal(0, he_std, (1, 1, 1, self.out_channels))
    
    def set_training(self, is_training):
        """设置是否为训练模式"""
        self.is_training = is_training
    
    def _pad_input(self, input_data):
        """
        对输入数据进行填充
        
        Args:
            input_data: 输入数据，形状为 (batch_size, height, width, channels)
            
        Returns:
            填充后的数据
        """
        if self.padding == 0:
            return input_data
        
        batch_size, height, width, channels = input_data.shape
        padded_height = height + 2 * self.padding
        padded_width = width + 2 * self.padding
        
        # 创建填充后的数组，初始化为0
        padded_data = np.zeros((batch_size, padded_height, padded_width, channels))
        
        # 将原始数据复制到填充后的中心区域
        padded_data[:, self.padding:self.padding+height, self.padding:self.padding+width, :] = input_data
        
        return padded_data
        
    def img2col(self, x, filter_size, stride):
        """
        将输入特征图转换为列矩阵，便于矩阵乘法实现卷积操作
        
        Args:
            x: 输入数据，形状为 (batch_size, height, width, channels)
            filter_size: 卷积核大小
            stride: 步长
            
        Returns:
            转换后的列矩阵
        """
        batch_size, height, width, channels = x.shape
        # 计算输出特征图的尺寸
        output_height = (height - filter_size) // stride + 1
        output_width = (width - filter_size) // stride + 1
        output_size = output_height * output_width
        
        # 初始化结果矩阵
        x_cols = np.zeros((output_size * batch_size, filter_size * filter_size * channels))
        
        # 构建列矩阵
        for b in range(batch_size):
            for h in range(0, height - filter_size + 1, stride):
                h_idx = h // stride
                for w in range(0, width - filter_size + 1, stride):
                    w_idx = w // stride
                    idx = h_idx * output_width + w_idx
                    window = x[b, h:h+filter_size, w:w+filter_size, :]
                    x_cols[b * output_size + idx, :] = window.reshape(-1)
        
        return x_cols
    
    def forward(self, input_data):
        """
        前向传播（新实现，使用矩阵乘法）
        
        Args:
            input_data: 输入数据，形状为 (batch_size, height, width, channels)
            
        Returns:
            输出特征图
        """
        # 保存原始输入，用于反向传播
        self.input_data = input_data
        self.input_shape = input_data.shape
        
        # NHWC -> NCHW（为了与参考实现保持一致的数据布局）
        batch_size, height, width, channels = input_data.shape
        x = np.transpose(input_data, (0, 3, 1, 2))
        
        # 对输入数据进行填充
        if self.padding > 0:
            p = self.padding
            x = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), "constant")
        
        # 保存填充后的输入
        self.padded_input = np.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        
        # 计算输出特征图的尺寸
        N, C, H, W = x.shape
        out_height = (H - self.kernel_size) // self.stride + 1
        out_width = (W - self.kernel_size) // self.stride + 1
        
        # 将卷积核权重reshape为列向量，便于矩阵乘法
        # 将权重从HWIO格式转换为OIHW格式，然后reshape为(out_channels, in_channels*kernel_size*kernel_size)
        weight_reshaped = np.transpose(self.weights, (3, 2, 0, 1)).reshape(self.out_channels, -1)
        
        # 将输入数据转换为列矩阵
        x_cols = self.img2col(self.padded_input, self.kernel_size, self.stride)
        self.x_cols = x_cols  # 保存用于反向传播
        
        # 执行矩阵乘法实现卷积
        result = np.dot(x_cols, weight_reshaped.T)
        # 添加偏置
        for c_out in range(self.out_channels):
            result[:, c_out] += self.bias[0, 0, 0, c_out]
        
        # 重新整形为输出特征图的形状(batch_size, out_height, out_width, out_channels)
        output = result.reshape(batch_size, out_height, out_width, self.out_channels)
        self.output_shape = output.shape
        
        return output

    """
    def forward(self, input_data):
        self.input_data = input_data
        self.input_shape = input_data.shape
        
        # 对输入数据进行填充
        self.padded_input = self._pad_input(input_data)
        
        batch_size, padded_height, padded_width, in_channels = self.padded_input.shape
        
        # 计算输出特征图的尺寸
        out_height = (padded_height - self.kernel_size) // self.stride + 1
        out_width = (padded_width - self.kernel_size) // self.stride + 1
        self.output_shape = (batch_size, out_height, out_width, self.out_channels)
        
        # 初始化输出数组
        output = np.zeros(self.output_shape)
        
        # 执行卷积操作
        for b in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    for c_out in range(self.out_channels):
                        # 当前滑动窗口的起始位置
                        h_start = h * self.stride
                        w_start = w * self.stride
                        
                        # 提取当前窗口的输入数据
                        window = self.padded_input[b, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size, :]
                        
                        # 计算卷积操作 (点积 + 偏置)
                        output[b, h, w, c_out] = np.sum(window * self.weights[:, :, :, c_out]) + self.bias[0, 0, 0, c_out]
        
        return output
    """

    def backward(self, grad_output):
        """
        反向传播（新实现，使用矩阵乘法）
        
        Args:
            grad_output: 从后一层传来的梯度, 形状为 (batch_size, out_height, out_width, out_channels)
            
        Returns:
            传递给前一层的梯度
        """
        batch_size, out_height, out_width, out_channels = grad_output.shape
        
        # 将梯度重塑为 (batch_size * out_height * out_width, out_channels)
        grad_reshaped = grad_output.reshape(-1, out_channels)
        
        # 计算卷积权重的梯度
        dw = np.dot(grad_reshaped.T, self.x_cols).reshape(out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        # 将梯度从OIHW格式转换回HWIO格式
        dw = np.transpose(dw, (2, 3, 1, 0))
        
        # 计算偏置的梯度
        db = np.sum(grad_reshaped, axis=0).reshape(1, 1, 1, out_channels)
        
        # 计算输入梯度
        # 将卷积核权重reshape为列向量，便于矩阵乘法
        weight_reshaped = np.transpose(self.weights, (3, 2, 0, 1)).reshape(self.out_channels, -1)
        
        # 计算输入特征图的梯度
        dx_cols = np.dot(grad_reshaped, weight_reshaped)
        
        # 将列矩阵重塑回特征图形状
        dx = np.zeros_like(self.padded_input)
        _, padded_height, padded_width, _ = self.padded_input.shape
        
        # 反向传播到输入
        for b in range(batch_size):
            for h in range(0, padded_height - self.kernel_size + 1, self.stride):
                h_idx = h // self.stride
                for w in range(0, padded_width - self.kernel_size + 1, self.stride):
                    w_idx = w // self.stride
                    idx = h_idx * out_width + w_idx
                    dx_window = dx_cols[b * out_height * out_width + idx, :].reshape(self.kernel_size, self.kernel_size, self.in_channels)
                    dx[b, h:h+self.kernel_size, w:w+self.kernel_size, :] += dx_window
        
        # 如果有填充，去除填充部分
        if self.padding > 0:
            _, in_height, in_width, _ = self.input_shape
            dx_input = dx[:, self.padding:self.padding+in_height, self.padding:self.padding+in_width, :]
        else:
            dx_input = dx
            
        # 更新梯度
        self.delta_weights = -dw / batch_size
        self.delta_bias = -db / batch_size
        
        return dx_input

    """
    def backward(self, grad_output):
        batch_size, out_height, out_width, out_channels = grad_output.shape
        
        # 初始化填充输入的梯度
        grad_padded_input = np.zeros_like(self.padded_input)
        
        # 计算权重和偏置梯度
        for b in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    for c_out in range(self.out_channels):
                        # 当前梯度值
                        grad_val = grad_output[b, h, w, c_out]
                        
                        # 当前窗口的起始位置
                        h_start = h * self.stride
                        w_start = w * self.stride
                        
                        # 更新权重梯度
                        self.delta_weights[:, :, :, c_out] -= self.padded_input[b, h_start:h_start+self.kernel_size, 
                                                                         w_start:w_start+self.kernel_size, :] * grad_val
                        
                        # 更新填充输入梯度（用于反向传播）
                        grad_padded_input[b, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size, :] -= \
                            self.weights[:, :, :, c_out] * grad_val
                        
                        # 更新偏置梯度
                        self.delta_bias[0, 0, 0, c_out] -= grad_val
        
        # 对偏置梯度进行平均
        self.delta_bias /= batch_size
        # 对权重梯度进行平均
        self.delta_weights /= batch_size
        
        # 如果有填充，需要从填充后的梯度中提取原始输入的梯度
        if self.padding > 0:
            _, in_height, in_width, _ = self.input_shape
            grad_input = grad_padded_input[:, self.padding:self.padding+in_height, self.padding:self.padding+in_width, :]
        else:
            grad_input = grad_padded_input
        
        return grad_input
    """

    def update(self, lr):
        """
        更新权重和偏置
        
        Args:
            lr: 学习率
        """
        self.weights += lr * self.delta_weights
        self.bias += lr * self.delta_bias
        
        # 清零梯度
        self.delta_weights = np.zeros_like(self.weights)
        self.delta_bias = np.zeros_like(self.bias)
    
    def get_weights(self):
        return self.weights
    
    def get_bias(self):
        return self.bias
    
    def set_weights(self, weights):
        self.weights = weights
    
    def set_bias(self, bias):
        self.bias = bias

class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        """
        初始化最大池化层
        
        Args:
            pool_size: 池化核大小
            stride: 步长
        """
        self.pool_size = pool_size
        self.stride = stride
        self.mask = None
        self.input_shape = None
        self.output_shape = None
        self.is_training = True
    
    def set_training(self, is_training):
        """设置是否为训练模式"""
        self.is_training = is_training
    
    def forward(self, input_data):
        """
        前向传播
        
        Args:
            input_data: 输入数据，形状为 (batch_size, height, width, channels)
            
        Returns:
            池化后的输出
        """
        self.input_shape = input_data.shape
        batch_size, in_height, in_width, in_channels = input_data.shape
        
        # 计算输出尺寸
        out_height = (in_height - self.pool_size) // self.stride + 1
        out_width = (in_width - self.pool_size) // self.stride + 1
        self.output_shape = (batch_size, out_height, out_width, in_channels)
        
        # 初始化输出和记录最大值位置的掩码
        output = np.zeros(self.output_shape)
        self.mask = np.zeros(self.input_shape)
        
        # 执行最大池化
        for b in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    for c in range(in_channels):
                        # 当前窗口的起始位置
                        h_start = h * self.stride
                        w_start = w * self.stride
                        
                        # 提取当前窗口
                        window = input_data[b, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size, c]
                        
                        # 找到最大值
                        max_value = np.max(window)
                        output[b, h, w, c] = max_value
                        
                        # 记录最大值的位置
                        # 注意：这种实现方式支持多个位置具有相同最大值的情况
                        max_indices = np.where(window == max_value)
                        # 将最大值位置标记为1（用于反向传播）
                        self.mask[b, h_start + max_indices[0][0], w_start + max_indices[1][0], c] = 1
        
        return output
    
    def backward(self, grad_output):
        """
        反向传播
        
        Args:
            grad_output: 从后一层传来的梯度
            
        Returns:
            传递给前一层的梯度
        """
        # 初始化输入梯度
        grad_input = np.zeros(self.input_shape)
        batch_size, out_height, out_width, in_channels = grad_output.shape
        
        # 反向传播梯度
        for b in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    for c in range(in_channels):
                        # 当前梯度值
                        grad_val = grad_output[b, h, w, c]
                        
                        # 当前窗口的起始位置
                        h_start = h * self.stride
                        w_start = w * self.stride
                        
                        # 只将梯度传递给最大值位置
                        # 使用掩码定位之前记录的最大值位置
                        window_mask = self.mask[b, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size, c]
                        grad_input[b, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size, c] += window_mask * grad_val
        
        return grad_input


class Flatten:
    def __init__(self):
        """
        初始化Flatten层，将输入展平为2D张量
        """
        self.input_shape = None
        self.is_training = True
    
    def set_training(self, is_training):
        """设置是否为训练模式"""
        self.is_training = is_training
    
    def forward(self, input_data):
        """
        前向传播
        
        Args:
            input_data: 输入数据，形状为 (batch_size, height, width, channels)
            
        Returns:
            展平后的输出，形状为 (batch_size, height*width*channels)
        """
        self.input_shape = input_data.shape
        batch_size = input_data.shape[0]
        # print("input:", self.input_shape)
        # output = input_data.reshape(batch_size, -1)
        # print("output:", output.shape)
        return input_data.reshape(batch_size, -1)
    
    def backward(self, grad_output):
        """
        反向传播
        
        Args:
            grad_output: 从后一层传来的梯度
            
        Returns:
            传递给前一层的梯度
        """
        return grad_output.reshape(self.input_shape)


class Dropout:
    def __init__(self, dropout_rate=0.0):
        """
        初始化Dropout层
        
        Args:
            dropout_rate: dropout概率，表示神经元被丢弃的概率
        """
        self.dropout_rate = dropout_rate
        self.mask = None
        self.is_training = True
    
    def set_training(self, is_training):
        """设置是否为训练模式"""
        self.is_training = is_training
    
    def forward(self, input_data):
        """
        前向传播，应用dropout
        
        Args:
            input_data: 输入数据
            
        Returns:
            应用dropout后的数据
        """
        if self.is_training and self.dropout_rate > 0:
            # 生成随机掩码，如果随机值小于p则丢弃(置为0)
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=input_data.shape)
            # 缩放未丢弃的神经元输出，保持期望值不变
            return input_data * self.mask / (1 - self.dropout_rate)
        else:
            # 测试阶段不使用dropout
            return input_data
    
    def backward(self, delta_output):
        """
        反向传播，传递梯度
        
        Args:
            delta_output: 上一层传来的梯度
            
        Returns:
            经过dropout层的梯度
        """
        if self.is_training and self.dropout_rate > 0:
            # 反向传播时应用相同的掩码
            return delta_output * self.mask / (1 - self.dropout_rate)
        else:
            return delta_output


class FullyConnected:
    def __init__(self, input_size, output_size, init_method=3, random_range=0.15, output_layer=False):
        """
        初始化全连接层
        
        Args:
            input_size: 输入大小
            output_size: 输出大小
            init_method: 初始化方法
            random_range: 随机初始化的范围或标准差
            output_layer: 是否为输出层
        """
        self.input_size = input_size
        self.output_size = output_size
        self.output_layer = output_layer
        
        # 初始化权重和偏置
        self._init_weights(init_method, random_range)
        
        # 存储中间结果
        self.input_data = None
        self.linear_output = None
        self.output_data = None
        
        # 存储梯度
        self.delta_weights = np.zeros_like(self.weights)
        self.delta_bias = np.zeros_like(self.bias)
        
        # 训练模式标志
        self.is_training = True
    
    def _init_weights(self, init_method, random_range):
        """初始化权重和偏置"""
        if init_method == 0:
            # 零初始化
            self.weights = np.zeros((self.input_size, self.output_size))
            self.bias = np.zeros((1, self.output_size))
        elif init_method == 1:
            # 均匀随机初始化
            self.weights = np.random.uniform(-random_range, random_range, (self.input_size, self.output_size))
            self.bias = np.random.uniform(-random_range, random_range, (1, self.output_size))
        elif init_method == 2:
            # 正态分布初始化
            self.weights = np.random.normal(0, random_range, (self.input_size, self.output_size))
            self.bias = np.random.normal(0, random_range, (1, self.output_size))
        else:
            # 何凯明初始化
            he_std = np.sqrt(2.0 / self.input_size)
            self.weights = np.random.normal(0, he_std, (self.input_size, self.output_size))
            self.bias = np.random.normal(0, he_std, (1, self.output_size))
    
    def set_training(self, is_training):
        """设置是否为训练模式"""
        self.is_training = is_training
    
    def forward(self, input_data, activation_function):
        """
        前向传播
        
        Args:
            input_data: 输入数据，形状为 (batch_size, input_size)
            activation_function: 激活函数
            
        Returns:
            层的输出
        """
        self.input_data = input_data
        # batch_size = input_data.shape[0]
        # print(input_data.shape)
        # 线性变换: y = Wx + b
        self.linear_output = np.matmul(input_data, self.weights) + self.bias
        
        # 应用激活函数
        if self.output_layer:
            # 输出层使用softmax激活函数进行多分类
            self.output_data = my_utils.softmax(self.linear_output)
        else:
            # 中间层使用指定的激活函数
                self.output_data = activation_function(self.linear_output)
        
        return self.output_data
    
    def backward(self, grade, activation_derivation):
        """
        反向传播
        
        Args:
            grade: 从后一层传来的梯度
            activation_derivation: 激活函数的导数
            
        Returns:
            传递给前一层的梯度
        """
        batch_size = self.input_data.shape[0]
        
        if self.output_layer:
            # 输出层损失对linear_data的求导结果
            delta_linear = -grade  # grade = true - pred
        else:
            # 非输出层，需要计算激活函数的导数
            delta_linear = grade * activation_derivation(self.output_data)
        
        # 计算权重和偏置的梯度
        weight_gradient = np.matmul(self.input_data.T, delta_linear) / batch_size
        bias_gradient = np.mean(delta_linear, axis=0, keepdims=True)
        
        # 更新参数的累计梯度
        self.delta_weights -= weight_gradient
        self.delta_bias -= bias_gradient
        
        # 计算传递给前一层的梯度
        backward_grade = np.matmul(delta_linear, self.weights.T)
        
        return backward_grade
    
    def update(self, lr):
        """
        更新权重和偏置
        
        Args:
            lr: 学习率
        """
        self.weights += lr * self.delta_weights
        self.bias += lr * self.delta_bias
        
        # 清零梯度
        self.delta_weights = np.zeros_like(self.weights)
        self.delta_bias = np.zeros_like(self.bias)
    
    def get_weights(self):
        return self.weights
    
    def get_bias(self):
        return self.bias
    
    def set_weights(self, weights):
        self.weights = weights
    
    def set_bias(self, bias):
        self.bias = bias 