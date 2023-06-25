import os
import pickle
import torch

import random
def read_data(file_path):
    radhar_data = []
    data = {}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            if line == '---':
                radhar_data.append(data)
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

    return radhar_data


def fill_frame(frame_data, target_length):
    frame_length = len(frame_data)
    if frame_length >= target_length:
        return frame_data[:target_length]

    filled_frame = frame_data.copy()
    for _ in range(target_length - frame_length):
        random_index = random.randint(0, frame_length - 1)
        random_point = frame_data[random_index]
        filled_frame.append(random_point)

    return filled_frame

def random_fill_frame(frame_data, target_length):
    frame_length = len(frame_data)
    if frame_length >= target_length:
        return frame_data[:target_length]

    filled_frame = frame_data.copy()
    for _ in range(target_length - frame_length):
        random_point = [random.random(), random.random(), random.random()]  # 随机生成一个点
        filled_frame.append(random_point)

    return filled_frame


def last_frame_fill(frame_data, target_length):
    frame_length = len(frame_data)
    if frame_length >= target_length:
        return frame_data[:target_length]

    filled_frame = frame_data.copy()
    last_point = frame_data[-1]
    for _ in range(target_length - frame_length):
        filled_frame.append(last_point)

    return filled_frame

def process_data(data_dir, classes):
    frames_per_class = 200
    frame_length = 40

    frames = []
    for class_file in classes:
        file_path = os.path.join(data_dir, class_file)
        class_data = read_data(file_path)

        for _ in range(frames_per_class):
            frame_data = []
            for data in class_data:
                frame_data.append([data.get('x', 0), data.get('y', 0), data.get('z', 0)])

            filled_frame = fill_frame(frame_data, frame_length)
            frames.append(filled_frame)

    # 转换为 PyTorch 张量
    frames_tensor = torch.tensor(frames, dtype=torch.float32)

    # 调整张量形状为 72*200*40*3
    frames_tensor = frames_tensor.view(72, frames_per_class, frame_length, 3)

    return frames_tensor



data_dir = 'E:/Code/Python/Project1/train'
file_prefix = 'train'
file_extension = '.txt'
num_files = 72

file_names = [file_prefix + str(i) + file_extension for i in range(1, num_files + 1)]
traindata = process_data(data_dir, file_names)

# 保存 traindata
with open('traindata.pkl', 'wb') as file:
    pickle.dump(traindata, file)
# print(traindata[0][0])