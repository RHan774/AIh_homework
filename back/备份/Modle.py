import numpy as np
import os
import my_utils

class MP(object):
    def __init__(self, task, batch_size, input_size, output_size, random_range, output_layer=False):
        """
        Args:
            task: 回归or多分类
            batch_size: 一个batch的大小
            input_size: 输入数据大小
            output_size: 输出数据大小
            random_range: 初始化参数时的标准差
            output_layer: 是否为神经网络最后一层. Defaults to False.
        """
        self.task = task
        self.batch_size = batch_size
        self.input_data = np.zeros((batch_size, input_size)) # 输入数据x
        random_range = np.sqrt(2.0 / (input_size + output_size)) # ------------------debug修改
        self.weight = np.random.normal(0., random_range, (input_size, output_size)) # 权重W
        self.bias = np.random.normal(0., random_range, (1, output_size))-0.1   # 偏置b
        self.linear_data = np.zeros((batch_size, output_size))   # Wx+b
        self.output_data = np.zeros_like(self.linear_data)  # 输出数据o
        self.output_layer = output_layer
        
        self.delta_weight = np.zeros_like(self.weight)
        self.delta_bias = np.zeros_like(self.bias)

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
        
    def backward(self, loss, activation_derivation):
        if self.output_layer:
            # 回归输出层和分类输出层对linear_data求导结果
            delta_output = -loss    # loss = true - pred
        else:
            # 非输出层，需要计算激活函数的导数
            delta_output = loss * activation_derivation(self.output_data)   # loss是上一层传下来的
        
        # 计算权重和偏置的梯度
        # input_data: (batch_size, input_size)
        # delta_output: (batch_size, output_size)
        weight_gradient = np.matmul(self.input_data.T, delta_output) / self.batch_size
        bias_gradient = np.mean(delta_output, axis=0, keepdims=True)
        
        # 更新参数的累计梯度
        self.delta_weight -= weight_gradient
        self.delta_bias -= bias_gradient
        
        # 计算传递给前一层的误差
        # weight: (input_size, output_size)
        # delta_output: (batch_size, output_size)
        # backward_loss应为 (batch_size, input_size)
        backward_loss = np.matmul(delta_output, self.weight.T)
        
        return backward_loss
    
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
    def __init__(self, task, layer_arch, lr, batch_size, random_range, activation_function,
                 activation_derivation, is_load=False, model_path=''):
        """
        Args:
            task: 任务类别(Regression / Classifier)
            layer_arch: 神经网络架构
            lr: 学习率
            batch_size: 一个batch中的数据数量
            random_range: 初始化参数的标准差
            activation_function: 激活函数
            activation_derivation: 激活函数的导数
            is_load: 是否加载已有模型. Defaults to False.
            model_path: 已有模型的地址. Defaults to ''.
        """
        self.task = task
        self.layer_arch = layer_arch
        self.lr = lr
        self.batch_size = batch_size
        self.random_range = random_range
        self.activation_function = activation_function
        self.activation_derivation = activation_derivation
        self.is_load = is_load
        self.load_path = model_path
        
        self.layers = []
        last_index = len(layer_arch) - 1
        # 中间层
        for hide_index in range(1, last_index):
            # task, batch_size, input_size, output_size, random_range, output_layer
            self.layers.append(MP(self.task, self.batch_size, self.layer_arch[hide_index-1],
                                  self.layer_arch[hide_index], self.random_range, False))
        # 输出层
        self.layers.append(MP(self.task, self.batch_size, self.layer_arch[last_index-1],
                              self.layer_arch[last_index], self.random_range, True))
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
        output = self.forward(batch_data)   # 前向传播
        loss = batch_results - output   # 计算损失（和损失函数无关）
        self.backward(loss) # 反向传播
        self.update()    # 更新参数
        
        return loss
    
    def val(self, val_data, val_results):
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
        
    def forward(self, input_data):
        for layer in self.layers:
            # 一层层向前传播
            input_data = layer.forward(input_data, self.activation_function)
        return input_data
    
    def backward(self, loss):
        for layer in reversed(self.layers):
            # 一层层反向传播
            loss = layer.backward(loss, self.activation_derivation)
            
    def update(self):
        for layer in self.layers:
            layer.update(self.lr)

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
            
        