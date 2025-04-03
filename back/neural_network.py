import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # 初始化权重和偏置
        self.w1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, x):
        # 前向传播
        self.z1 = np.matmul(x, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.matmul(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, x, y):
        m = x.shape[0]
        
        # 计算输出层的误差
        dz2 = self.a2 - y
        dw2 = np.matmul(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # 计算隐藏层的误差
        dz1 = np.matmul(dz2, self.w2.T) * self.sigmoid_derivative(self.a1)
        dw1 = np.matmul(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # 更新权重和偏置
        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2
        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1
    
    def train(self, x, y, epochs=1000):
        losses = []
        for epoch in range(epochs):
            # 前向传播
            y_pred = self.forward(x)
            
            # 计算损失
            loss = np.mean((y - y_pred) ** 2)
            losses.append(loss)
            
            # 反向传播
            self.backward(x, y)
            
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}')
        
        return losses
    
    def predict(self, x):
        return self.forward(x)

def generate_regression_data():
    # 生成回归问题的数据
    X, y = make_regression(n_samples=1000, n_features=1, noise=0.1)
    y = y.reshape(-1, 1)
    return X, y

def generate_classification_data():
    # 生成分类问题的数据
    X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, n_clusters_per_class=1)
    y = y.reshape(-1, 1)
    return X, y

def plot_results(X, y, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, label='真实值')
    plt.plot(X, y_pred, 'r-', label='预测值')
    plt.title(title)
    plt.legend()
    plt.show()

def main():
    # 回归问题
    print("训练回归模型...")
    X_reg, y_reg = generate_regression_data()
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2)
    
    # 标准化数据
    scaler = StandardScaler()
    X_train_reg_scaled = scaler.fit_transform(X_train_reg)
    X_test_reg_scaled = scaler.transform(X_test_reg)
    
    # 创建和训练模型
    nn_reg = NeuralNetwork(input_size=1, hidden_size=4, output_size=1)
    losses_reg = nn_reg.train(X_train_reg_scaled, y_train_reg)
    
    # 预测和可视化
    y_pred_reg = nn_reg.predict(X_test_reg_scaled)
    plot_results(X_test_reg, y_test_reg, y_pred_reg, '回归问题结果')
    
    # 分类问题
    print("\n训练分类模型...")
    X_clf, y_clf = generate_classification_data()
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2)
    
    # 标准化数据
    X_train_clf_scaled = scaler.fit_transform(X_train_clf)
    X_test_clf_scaled = scaler.transform(X_test_clf)
    
    # 创建和训练模型
    nn_clf = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    losses_clf = nn_clf.train(X_train_clf_scaled, y_train_clf)
    
    # 预测和评估
    y_pred_clf = nn_clf.predict(X_test_clf_scaled)
    accuracy = np.mean((y_pred_clf > 0.5) == y_test_clf)
    print(f'分类准确率: {accuracy:.4f}')
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses_reg)
    plt.title('回归问题损失曲线')
    plt.subplot(1, 2, 2)
    plt.plot(losses_clf)
    plt.title('分类问题损失曲线')
    plt.show()

if __name__ == "__main__":
    main() 