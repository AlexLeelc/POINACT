import torch
import torch.nn as nn
import torch.optim as optim
from PointNet_lstm import HAR_model
from torch.optim.lr_scheduler import StepLR
# import matplotlib.pyplot as plt
# import matplotlib
# import numpy as np
# from data import process_data,radhar_data,test_data,radhar_data3
import pickle
import numpy as np

# 加载 traindata
with open('traindata.pkl', 'rb') as file:
    traindata = pickle.load(file)
print(traindata.size())
with open('testdata.pkl', 'rb') as file:
    testdata = pickle.load(file)
# 初始化模型和损失函数、优化器


# 定义模型和损失函数
model = HAR_model(frame_num=200, output_dim=5)  # 根据您的模型定义进行实例化，这里假设模型名称为HAR_model
criterion = nn.CrossEntropyLoss()

# 定义训练数据和真实结果
truthdata = torch.tensor([0]*13 + [1]*14 + [2]*13 + [3]*15 + [4]*17).reshape(72, 1)  # 替换为您的真实结果张量

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# optimizer = optim.SGD(model.parameters(), lr=0.0001)

# 训练模型
num_epochs = 50
model.train()
accuracies = []
for epoch in range(num_epochs):
    optimizer.zero_grad()

    output = model(traindata)
    loss = criterion(output, truthdata.squeeze(1))
    loss.backward()
    optimizer.step()

    _, predicted_labels = torch.max(output, 1)
    correct_predictions = (predicted_labels == truthdata.squeeze(1)).sum().item()
    accuracy = correct_predictions / traindata.shape[0]
    accuracies.append(accuracy)  # 将精度值添加到列表中

    print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss.item()))

filename = 'accuracies2.txt'
np.savetxt(filename, accuracies)

# torch.save(model.state_dict(), 'model.pth')
#------------------------------------------------------------
truthdata_test = torch.tensor([0]*13 + [1]*10 + [2]*11 + [3]*10 + [4]*14).reshape(58, 1)  # 替换为您的真实结果张量
# 测试模型
model.eval()  # 将模型设置为评估模式

with torch.no_grad():
    output_test = model(traindata)
    _, predicted_labels_test = torch.max(output_test, 1)
    correct_predictions_test = (predicted_labels_test == truthdata.squeeze(1)).sum().item()
    accuracy_test = correct_predictions_test / traindata.shape[0]

print('Test Accuracy: {:.2f}%'.format(accuracy_test * 100))


