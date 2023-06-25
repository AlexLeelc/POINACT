import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSizePolicy
from PyQt5.QtCore import QTimer
from PyQt5 import QtCore
from PyQt5.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Poinact系统 v1.0")
        self.setGeometry(100, 100, 800, 800)
        self.setWindowIcon(QIcon("E:/Code/Python/Project1/icon.png"))  # 设置应用程序图标

        # 创建按钮
        self.select_button = QPushButton("选择文件", self)
        # self.select_button.setGeometry(300, 10, 50, 30)  # 设置按钮的位置和大小 左部、顶部、宽度、高度
        self.select_button.clicked.connect(self.select_file)

        self.plot_button = QPushButton("绘制", self)
        self.plot_button.setGeometry(300, 50, 100, 30)
        self.plot_button.clicked.connect(self.plot_point_cloud)

        self.pause_button = QPushButton("暂停", self)
        self.pause_button.setGeometry(300, 90, 100, 30)
        self.pause_button.clicked.connect(self.pause_point_cloud)

        self.reset_button = QPushButton("复位", self)
        self.reset_button.setGeometry(300, 130, 100, 30)
        self.reset_button.clicked.connect(self.reset_point_cloud)

        self.Poinact_button = QPushButton("Poinact", self)
        self.Poinact_button.setGeometry(300, 170, 100, 30)
        self.Poinact_button.clicked.connect(self.Poinact)

        self.path_label = QLabel(self) # 展示文件路径
        self.path_label.setGeometry(300, 210, 100, 30)  # 设置路径标签的位置和大小
        self.path_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # 设置标签的大小策略为固定大小
        self.path_label.setFixedSize(1200, 30)  # 设置标签的固定大小
        self.path_label.setAlignment(QtCore.Qt.AlignCenter)

        self.class_label = QLabel(self) # 展示文件路径
        self.class_label.setGeometry(300, 250, 100, 30)  # 设置路径标签的位置和大小
        self.class_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # 设置标签的大小策略为固定大小
        self.class_label.setFixedSize(1200, 30)  # 设置标签的固定大小

        # 创建 Matplotlib 的图形对象
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        # 创建 FigureCanvas 对象，并将图形对象嵌入其中
        self.canvas = FigureCanvas(self.fig)

        # 创建一个 QVBoxLayout 布局管理器
        layout = QVBoxLayout()

        # 创建一个单独的部件来容纳绘图框
        plot_widget = QWidget()
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.canvas)
        plot_widget.setLayout(plot_layout)

        # 将按钮添加到布局管理器中
        layout.addWidget(self.select_button)
        layout.addWidget(self.plot_button)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.reset_button)
        layout.addWidget(self.Poinact_button)
        layout.addWidget(self.path_label)
        layout.addWidget(self.class_label)

        # 将绘图框部件添加到布局管理器中
        layout.addWidget(plot_widget)

        # 创建一个 QWidget 作为主窗口的中心部件
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # 初始化radhar_data
        self.radhar_data=[]

        # 初始化file_path
        self.file_path = None

        # 初始化画图的点云数据
        self.point_clouds = []

        # 初始化模型测试输入的点云数据
        self.test_data_list = []

        # 初始化模型测试输入的点云张量
        self.test_data_tensor = torch.empty(1, 200, 40, 3)

        # 初始化帧
        self.frame = 1
        self.last_frame = -1

    def select_file(self):
        self.test_data_tensor = torch.empty(1, 200, 40, 3)
        self.test_data_list = []
        self.point_clouds = []
        self.frame = 1
        self.last_frame = -1
        self.radhar_data=[]
        self.file_path = None
        options = QFileDialog.Options()
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Text Files (*.txt)")
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            self.file_path = selected_files[0]
            self.path_label.setText("文件路径：" + self.file_path)  # 更新路径标签的文本
        self.frame_process()
        self.process_data()
        # print(self.radhar_data[0])

    def plot_point_cloud(self):
        # 手动触发绘图
        # 创建定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_point_cloud)
        self.timer.start(250)  # 时间间隔为0.25秒（250毫秒）

    def update_point_cloud(self):

        if self.frame <= self.last_frame:
            # 生成指定帧的点云数据（一帧）
            frame_data = [point for point in self.radhar_data if point.get('frame') == self.frame]

            # 提取点云数据
            x = [point.get('x', 0) for point in frame_data]
            y = [point.get('y', 0) for point in frame_data]
            z = [point.get('z', 0) for point in frame_data]

            # 将一帧点云数据添加到列表中
            self.point_clouds.append((x, y, z))

            # 绘制当前帧的点云
            self.ax.clear()
            self.ax.scatter3D(x, y, z, s=100)

            # 更新图形
            self.canvas.draw()

            # 更新帧数
            self.frame += 1

            if self.frame > self.last_frame:
                # 绘制完成，停止定时器
                self.timer.stop()
        else:
            # 绘制完成，停止定时器
            self.timer.stop()

    def pause_point_cloud(self):
        # 暂停绘制
        self.timer.stop()

    def reset_point_cloud(self):
        self.frame = 1
        self.ax.clear()
        self.canvas.draw()

    def frame_process(self):
        # 处理radhar_data并加frame
        data = {}
        frame = 0  # 初始化帧数为1

        with open(self.file_path, 'r') as file:
            for line in file:
                line = line.strip()

                if line == '---':
                    data['frame'] = frame  # 将当前帧数保存到数据中
                    self.radhar_data.append(data)
                    if 'point_id' in data and data['point_id'] == 0:
                        data['frame'] = frame+1 # 将当前帧数保存到数据中
                        frame += 1  # 如果 point_id 是 0，则帧数加1
                    data = {}
                else:
                    if ':' not in line:
                        continue

                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()

                    if key == 'header' or key == 'stamp':
                        continue

                    try:
                        value = float(value)
                        if value.is_integer():
                            value = int(value)
                    except ValueError:
                        pass

                    data[key] = value

        # 获取最后一帧的frame
        if len(self.radhar_data) > 0:
            last_data = self.radhar_data[-1]
            last_frame = last_data.get('frame')
            if last_frame is not None:
                self.last_frame = last_frame
            else:
                print("Frame value not found.")
        else:
            print("No data available.")


    def process_data(self):
        # 只提取200帧
        last_frame = 200
        target_length = 40

        for frame in range(1, last_frame + 1):
            frame_data = [point for point in self.radhar_data if point.get('frame') == frame]

            points = [[point.get('x', 0), point.get('y', 0), point.get('z', 0)] for point in frame_data]

            if len(points) < target_length:
                # 从前面的数据中随机采样填充到当前帧的数据中
                real_point_number = len(points) # 记录真实点数
                while len(points) < target_length:
                    points.append(random.choice(points[0:real_point_number])) # 在0到real_point_number-1索引下随机获得xyz的数据

            self.test_data_list.append(points)
        # print(self.test_data_list[0])
        # print(len(self.test_data_list[0]))
        # print(len(self.test_data_list))
        # 转换为张量
        self.test_data_tensor = torch.tensor(self.test_data_list, dtype=torch.float32)
        # 增加一个维度
        self.test_data_tensor = self.test_data_tensor.unsqueeze(0)
        # 打印张量的形状
        # print(self.test_data_tensor[0][0])

        # 填充不足200帧的部分
        # while len(self.test_data_list) < last_frame:
        #     last_frame_data = self.test_data_list[-1]
        #     self.test_data_list.append(last_frame_data)
        #
        # # 将数据转换为NumPy数组并调整形状为200x40x3
        # self.test_data_list = np.array(self.test_data_list)
        # self.test_data_list = self.test_data_list.reshape(last_frame, target_length, 3)
        # print(self.test_data_list[0])
        # print(self.test_data_list.size())

        # frames = []
        # frame_data = []
        # frame_count = 0
        #
        # for data in self.radhar_data:
        #     frame_data.append([data.get('x', 0), data.get('y', 0), data.get('z', 0)])
        #
        #     if data.get('point_id') == 0:
        #         filled_frame = fill_frame(frame_data, 40)
        #         frames.append(filled_frame)
        #         frame_data = []
        #         frame_count += 1
        #
        #     if frame_count >= 200:
        #         break
        #
        # # 填充不足200帧的部分
        # while len(frames) < 200:
        #     last_frame = frames[-1]
        #     frames.append(last_frame)
        #
        # # 转换为PyTorch张量
        # frames_tensor = torch.tensor(frames, dtype=torch.float32)
        #
        # # 调整张量形状为1*200*40*3
        # frames_tensor = frames_tensor.unsqueeze(0)
        #
        # return frames_tensor

    def Poinact(self):
        # 创建一个新的模型对象
        model = HAR_model(frame_num=200, output_dim=5)

        # 加载保存的模型状态字典
        model.load_state_dict(torch.load('model.pth'))
        # print(self.test_data_tensor.size())

        # 将模型设置为评估模式（如果需要）
        model.eval()
        tensor = model(self.test_data_tensor)


        # 使用torch.argmax()函数找到最大值所在的索引
        max_index = torch.argmax(tensor)
        self.class_label.setAlignment(QtCore.Qt.AlignCenter)
        self.class_label.setText("是第" + str(max_index.item()+1) + "个类")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
