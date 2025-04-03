import numpy as np

# sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 前向传播
def forward(x, w, b):
    return sigmoid(np.matmul(x, w) + b)

# sigmoid 函数的导数
def sigmoid_derivative(x):
    return x * (1 - x)

# 拟合问题的损失函数
def loss_function_fit(y_true, y_pred):
    return ((y_true - y_pred) ** 2) / 2

# 分类问题的损失函数
def loss_function_classify(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred))

# 拟合函数的梯度下降
def backward_fit(x, y, w, b):




