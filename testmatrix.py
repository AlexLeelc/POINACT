import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib
import random
import torch
import os
import pickle
from PointNet_lstm import HAR_model
matplotlib.use('Qt5Agg')

with open('testdata.pkl', 'rb') as file:
    test_data_tensor = pickle.load(file)
model = HAR_model(frame_num=200, output_dim=5)

# 加载保存的模型状态字典
model.load_state_dict(torch.load('model.pth'))

# 将模型设置为评估模式（如果需要）
model.eval()

# 假设预测结果为一个72个5维张量的列表，每个张量表示一个样本的预测结果
predictions = model(test_data_tensor) # 填入你的预测结果列表

# 假设真实标签为一个长度为72的列表，每个元素表示对应样本的真实类别索引
truthdata = torch.tensor([0]*13 + [1]*10 + [2]*11 + [3]*10 + [4]*14).reshape(58, 1)  # 替换为您的真实结果张量

with torch.no_grad():
    output_test = model(test_data_tensor)
    _, predicted_labels_test = torch.max(output_test, 1)
    correct_predictions_test = (predicted_labels_test == truthdata.squeeze(1)).sum().item()
    accuracy_test = correct_predictions_test / test_data_tensor.shape[0]

print('Test Accuracy: {:.2f}%'.format(accuracy_test * 100))


# 初始化混淆矩阵
confusion_matrix = np.zeros((5, 5), dtype=int)

# 遍历每个样本的预测结果和真实标签
for i in range(len(predictions)):
    # 获取预测结果和真实标签
    prediction = predictions[i]
    true_label = truthdata[i].item()

    # 找到预测结果中最大值所在的索引，即指代的类别
    predicted_class = torch.argmax(prediction).item()

    # 更新混淆矩阵
    confusion_matrix[true_label][predicted_class] += 1


row_totals = confusion_matrix.sum(axis=1)  # 每一行的总样本数量

# 将混淆矩阵每一行除以对应的总样本数量，并乘以100，得到百分比形式
confusion_matrix_percentage = np.round((confusion_matrix / row_totals[:, np.newaxis]) * 100, 2)

# 打印每一行的百分比形式的混淆矩阵
print(confusion_matrix_percentage)

# 定义类别标签
class_labels = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']

# 定义混淆矩阵数据（示例）

# 绘制混淆矩阵图
plt.imshow(confusion_matrix_percentage, cmap='Blues')

# 添加颜色条
plt.colorbar()

# 设置坐标轴标签
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# 设置类别标签刻度
plt.xticks(np.arange(len(class_labels)), class_labels)
plt.yticks(np.arange(len(class_labels)), class_labels)

# 在热图上显示数值
for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        plt.text(j, i, confusion_matrix_percentage[i, j], ha='center', va='center', color='white')

# 调整图像布局
plt.tight_layout()

# 显示图像
plt.show()