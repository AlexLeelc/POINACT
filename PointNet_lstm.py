import torch.nn as nn
import torch



from TimeDistributed import TimeDistributed
from PointNet import PointNetfeat


class Sub_PointNet(nn.Module):# 继承自 nn.Module 的类，这里会自动处理输入的数据，会自动省略1那个维度
    def __init__(self):
        super(Sub_PointNet, self).__init__()
        self.pointnet = PointNetfeat(global_feat = True)

    def forward(self,x):
        # torch.Size([60, 40, 3])
        x = x.permute(0, 2, 1)# 将原有的维度2放到1，维度1放到2
        out,_,_ = self.pointnet(x)# 提取的特征 out，以及其他未使用的返回值 _
        return out # 输出的是还未分类的结果


class HAR_model(nn.Module):
    def __init__(self, output_dim, frame_num):
        super(HAR_model, self).__init__()
        # 1st layer group
        self.pointnet = TimeDistributed(nn.Sequential(Sub_PointNet()))# 将 Sub_PointNet 包装在 nn.Sequential 中，可以将其作为整个网络的一部分进行处理。
        # 使用 nn.Sequential 的好处是简化了模型的定义和管理。它提供了一种简洁的方式来组织和串联各个模块，避免了手动定义多个 forward 方法的麻烦
        # 使用 TimeDistributed 将 nn.Sequential(Sub_PointNet()) 进一步包装，可以将 Sub_PointNet 应用于输入数据的每个时间步，以实现对时间序列数据的处理
        # embedding_dim, hidden_size, num_layers
        self.lstm_net = nn.LSTM(1024, 16,num_layers=1, dropout=0,bidirectional=True)
        # 每个时间步的输入特征维度为 1024；LSTM模型在每个时间步会输出一个大小为16的隐藏状态，隐藏状态用于存储模型在当前时间步的记忆信息和学习到的特征；只有一个LSTM层；双向LSTM会在每个时间步同时处理正向和反向的输入序列，从而更好地捕捉上下文信息
        self.dense = nn.Sequential( # 将上述三个层按照顺序组合起来
            nn.Linear(frame_num*32,output_dim),# 全链接，输入frame_num*32，输出output_dim维
            nn.Dropout(0.2),
            nn.Softmax(dim=1),# Softmax 计算多类别分类问题的概率
            )

    def forward(self, data):
        data = self.pointnet(data)
        # torch.Size([72, 200, 1024])



        data = data.permute(1,0,2)
        # input(seq_length, batch_size, input_size) [frame, N, 1024] 将数据调整为200 72 1024进行输入
        data,hn = self.lstm_net(data)
        data = data.permute(1,0,2)


        data = data.reshape(data.size(0),-1)
        return self.dense(data)


# if __name__ == '__main__':
#
#
#     a = torch.randn(1,60,40,3)# 32帧 一帧有40个点
#     model = HAR_model(frame_num=60,output_dim=5)
#     print(model(a))
#     print(model(a).shape)