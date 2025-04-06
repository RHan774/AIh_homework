import numpy as np
import os
import my_utils

class Dropout(object):
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

class MP(object):
    def __init__(self, task, batch_size, input_size, output_size, init_method, random_range, output_layer=False):
        """
        Args:
            task: 回归or多分类
            batch_size: 一个batch的大小
            input_size: 输入数据大小
            output_size: 输出数据大小
            init_method: 初始化权重的方法
            random_range: 初始化参数时的范围或标准差
            output_layer: 是否为神经网络最后一层. Defaults to False.
        """
        self.task = task
        self.batch_size = batch_size
        self.input_data = np.zeros((batch_size, input_size)) # 输入数据x
        self.is_training = True  # 标记是否为训练模式
        
        # 使用不同的初始化方法
        if init_method == 0:
            # 零初始化
            self.weight = np.zeros((input_size, output_size))
            self.bias = np.zeros((1, output_size))
        elif init_method == 1:
            # 均匀随机初始化
            self.weight = np.random.uniform(-random_range, random_range, (input_size, output_size))
            self.bias = np.random.uniform(-random_range, random_range, (1, output_size)) - 1    # 将bias先置为负数
        elif init_method == 2:
            # 正态分布初始化
            self.weight = np.random.normal(0., random_range, (input_size, output_size))
            self.bias = np.random.normal(0., random_range, (1, output_size)) - 1    # 将bias先置为负数
        elif init_method == 3:
            # He初始化（适用于ReLU激活函数）
            he_std = np.sqrt(2.0 / input_size)
            self.weight = np.random.normal(0., he_std, (input_size, output_size))
            self.bias = np.random.normal(0., he_std, (1, output_size))
        elif init_method == 4:
            # 稀疏初始化
            # 每个神经元随机连接sqrt(n)个输入神经元
            self.weight = np.zeros((input_size, output_size))
            connections_per_neuron = int(np.sqrt(input_size))
            for i in range(output_size):
                # 随机选择连接的输入神经元
                indices = np.random.choice(input_size, connections_per_neuron, replace=False)
                self.weight[indices, i] = np.random.normal(0., 0.01, connections_per_neuron)
            self.bias = np.zeros((1, output_size))
        else:
            # Xavier/Glorot初始化
            if init_method == 5 or init_method == 6:
                xavier_range = np.sqrt(6.0 / (input_size + output_size))
            elif init_method == 7 or init_method == 8:
                xavier_range = np.sqrt(6.0 / (input_size + output_size)) * np.sqrt(2)
            else:
                xavier_range = np.sqrt(6.0 / (input_size + output_size)) * 4
            if init_method == 5 or init_method == 7 or init_method == 9:
                self.weight = np.random.uniform(-xavier_range, xavier_range, (input_size, output_size))
                self.bias = np.random.uniform(-xavier_range, xavier_range, (1, output_size)) - 1    # 将bias先置为负数
            else:
                self.weight = np.random.normal(0., xavier_range, (input_size, output_size))
                self.bias = np.random.normal(0., xavier_range, (1, output_size)) - 1    # 将bias先置为负数
            
        self.linear_data = np.zeros((batch_size, output_size))   # Wx+b
        self.output_data = np.zeros_like(self.linear_data)  # 输出数据o
        self.output_layer = output_layer
        
        self.delta_weight = np.zeros_like(self.weight)
        self.delta_bias = np.zeros_like(self.bias)
    
    def set_training(self, is_training):
        """设置是否为训练模式"""
        self.is_training = is_training

    def forward(self, input_data, activation_function):
        self.input_data = input_data.reshape((self.batch_size, -1))
            
        # 对输入做线性运算：Wx + b
        self.linear_data = np.matmul(self.input_data, self.weight) + self.bias
        # 对线性运算结果执行激活函数（或不执行），得到输出
        if self.output_layer == True:
            if self.task == "Regression":
                self.output_data = self.linear_data # 回归问题的最后一层直接输出
            else:
                self.output_data = my_utils.softmax(self.linear_data)   # 分类问题的最后一层进行softmax
        else:
            self.output_data = activation_function(self.linear_data)    # 对中间层调用激活函数
        return self.output_data
        
    def backward(self, grade, activation_derivation):
        if self.output_layer:
            # 回归输出层和分类输出层对linear_data求导结果
            delta_linear_output = -grade    # grade = true - pred
        else:
            # 非输出层，需要计算激活函数的导数
            delta_linear_output = grade * activation_derivation(self.output_data)   # grade是上一层传下来的

        
        # 计算权重和偏置的梯度
        # input_data: (batch_size, input_size)
        # delta_linear_output: (batch_size, output_size)
        weight_gradient = np.matmul(self.input_data.T, delta_linear_output) / self.batch_size
        bias_gradient = np.mean(delta_linear_output, axis=0, keepdims=True)
        
        # 更新参数的累计梯度
        self.delta_weight -= weight_gradient
        self.delta_bias -= bias_gradient
        
        # 计算传递给前一层的梯度
        # weight: (input_size, output_size)
        # delta_linear_output: (batch_size, output_size)
        # backward_loss应为 (batch_size, input_size)
        backward_grade = np.matmul(delta_linear_output, self.weight.T)
            
        return backward_grade
    
    def update(self, lr):
        self.weight += lr * self.delta_weight
        self.bias += lr * self.delta_bias
        # 一个batch执行完，进行梯度清零
        self.delta_weight = np.zeros_like(self.weight)
        self.delta_bias = np.zeros_like(self.bias)

    def init_weight(self, w):
        self.weight = w

    def init_bias(self, b):
        self.bias = b

    def get_weight(self):
        return self.weight
    
    def get_bias(self):
        return self.bias


class MLP(object):
    def __init__(self, task, layer_arch, batch_size, lr=0.01, init_method=3, random_range=0.15, activation_function=0,
                 activation_derivation=0, use_dropout=False, dropout_rates=None, 
                 use_lr_scheduler=False, use_cosine_decay=False, use_warmup=False,
                 warmup_epochs=0, min_lr_ratio=0.01, total_epochs=1000,
                 is_load=False, model_path=''):
        """
        Args:
            task: 任务类别(Regression / Classifier)
            layer_arch: 神经网络架构
            batch_size: 一个batch中的数据数量
            lr: 初始学习率
            init_method: 初始化权重的方法
            random_range: 初始化参数的标准差或范围
            activation_function: 激活函数
            activation_derivation: 激活函数的导数
            use_dropout: 是否使用dropout
            dropout_rates: 每层的dropout概率列表，长度应该等于layer_arch的长度
            use_lr_scheduler: 是否使用学习率调度
            use_cosine_decay: 是否使用余弦衰减
            use_warmup: 是否使用学习率预热
            warmup_epochs: 预热的epoch数
            min_lr_ratio: 最小学习率与初始学习率的比例
            total_epochs: 总的训练epoch数
            is_load: 是否加载已有模型. Defaults to False.
            model_path: 已有模型的地址. Defaults to ''.
        """
        self.task = task
        self.layer_arch = layer_arch
        self.init_lr = lr  # 初始学习率
        self.lr = lr  # 当前学习率
        self.batch_size = batch_size
        self.init_method = init_method
        self.random_range = random_range
        self.activation_function = activation_function
        self.activation_derivation = activation_derivation
        self.use_dropout = use_dropout
        self.dropout_rates = dropout_rates if dropout_rates is not None else [0.0] * len(layer_arch)
        
        # 学习率调度相关参数
        self.use_lr_scheduler = use_lr_scheduler
        self.use_cosine_decay = use_cosine_decay
        self.use_warmup = use_warmup
        self.warmup_epochs = warmup_epochs
        self.min_lr_ratio = min_lr_ratio
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
        self.is_load = is_load
        self.load_path = model_path
        
        # 网络层列表
        self.layers = []
        # Dropout层列表
        self.dropouts = []
        
        last_index = len(layer_arch) - 1
        
        # 添加输入层dropout（处理输入数据）
        if self.use_dropout:
            self.dropouts.append(Dropout(self.dropout_rates[0]))
        
        # 中间层
        for hide_index in range(1, last_index):
            # task, batch_size, input_size, output_size, init_method, random_range, output_layer
            self.layers.append(MP(self.task, self.batch_size, self.layer_arch[hide_index-1],
                                  self.layer_arch[hide_index], self.init_method, self.random_range, False))
            # 每个隐藏层后面添加一个dropout层
            if self.use_dropout:
                self.dropouts.append(Dropout(self.dropout_rates[hide_index]))
        
        # 输出层
        self.layers.append(MP(self.task, self.batch_size, self.layer_arch[last_index-1],
                              self.layer_arch[last_index], self.init_method, self.random_range, True))
        
        # 加载已有模型数据
        if is_load:
            self.load_model()
    
    def train(self, batch_data, batch_results):
        """
        在一个batch上进行训练
        
        Args:
            batch_data: 一个batch的输入数据(batch_size*layer_arch[0])
            batch_results: 一个batch的标签(batch_size*layer_arch[最末])
        
        Returns:
            batch的损失值
        """
        # 设置所有层为训练模式
        self._set_training_mode(True)
        
        output = self.forward(batch_data)   # 前向传播
        loss = batch_results - output   # 计算损失（和损失函数无关）
        self.backward(loss) # 反向传播
        self.update()    # 更新参数
        
        return loss
    
    def train_epoch(self, train_data, train_labels):
        """
        训练一个epoch
        
        Args:
            train_data: 训练数据
            train_labels: 训练标签
        """
        # 更新学习率（如果使用学习率调度）
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
            # 打印当前epoch和学习率，便于调试
            if self.current_epoch % 10 == 0:
                print(f"Epoch {self.current_epoch}, Learning Rate: {self.lr:.6f}")
                
        # 训练一个epoch
        for batch_data, batch_labels in my_utils.get_batch(train_data, train_labels, self.batch_size, shuffle=True):
            self.train(batch_data, batch_labels)
            
        # 更新epoch计数
        self.current_epoch += 1
    
    def val(self, val_data, val_results):
        # 设置所有层为评估模式（不使用dropout）
        self._set_training_mode(False)
        
        correct = 0
        loss_sum = 0
        total = len(val_data)
        
        for data, results in my_utils.get_batch(val_data, val_results, self.batch_size, False):
            output = self.forward(data)
            
            if self.task == "Regression":
                # 计算均方误差
                loss_sum += np.mean((results - output) ** 2)
            else:
                # 计算交叉熵损失 (防止log(0)，添加一个小的epsilon值)
                epsilon = 1e-10
                loss_sum += -np.sum(results * np.log(output + epsilon))
                
                # 计算正确率
                result = np.argmax(results, axis=1)
                pred = np.argmax(output, axis=1)
                correct += np.sum(pred == result)
                
        avg_loss = loss_sum / total
        accu = correct / total
        
        if self.task == "Regression":
            return avg_loss
        else:
            return (avg_loss, accu)
    
    def _set_training_mode(self, is_training):
        """设置所有层的训练/评估模式"""
        for layer in self.layers:
            layer.set_training(is_training)
        
        for dropout in self.dropouts:
            dropout.set_training(is_training)
        
    def forward(self, input_data):
        # 首先应用输入层dropout（如果存在）
        if self.use_dropout:
            input_data = self.dropouts[0].forward(input_data)
        
        # 通过每一层
        for i, layer in enumerate(self.layers):
            # 前向传播通过当前层
            input_data = layer.forward(input_data, self.activation_function)
            # 如果不是最后一层且使用dropout，则应用dropout
            if i < len(self.layers) - 1 and self.use_dropout:
                input_data = self.dropouts[i+1].forward(input_data)
                
        return input_data
    
    def backward(self, loss):
        # 从最后一层开始反向传播
        for i in range(len(self.layers)-1, -1, -1):
            # 通过dropout层反向传播（如果存在且不是最后一层）
            if i < len(self.layers) - 1 and self.use_dropout:
                loss = self.dropouts[i+1].backward(loss)
            
            # 通过当前层反向传播
            loss = self.layers[i].backward(loss, self.activation_derivation)
            
        # 如果存在输入层dropout，则通过它反向传播
        if self.use_dropout:
            loss = self.dropouts[0].backward(loss)
            
    def update(self):
        for layer in self.layers:
            layer.update(self.lr)  # 使用当前学习率

    def load_model(self):
        print("--------------- loading existed model ---------------\n")
        index = 0
        for layer in self.layers:
            load_file_w = os.path.join(self.load_path, "w%d%d.npy"%(index, index+1))
            load_file_b = os.path.join(self.load_path, "b%d%d.npy"%(index, index+1))
            w = np.load(load_file_w)
            b = np.load(load_file_b)
            layer.init_weight(w)
            layer.init_bias(b)
            index += 1

    def save_model(self):
        print("------------------- saving  model -------------------\n")
        index = 0
        for layer in self.layers:
            save_file_w = os.path.join(self.load_path, "w%d%d.npy"%(index, index+1))
            save_file_b = os.path.join(self.load_path, "b%d%d.npy"%(index, index+1))
            w = layer.get_weight()
            b = layer.get_bias()
            np.save(save_file_w, w)
            np.save(save_file_b, b)
            index += 1
            
        