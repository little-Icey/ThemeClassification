import torch
import pandas as pd
import numpy as np
import torch.utils.data as Data
import torch.nn.functional as F
from torch.nn import init
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

maledata = pd.read_csv("../data/data_numpyM.csv")
femaledata = pd.read_csv("../data/data_numpyF.csv")
maledata.columns = ["f{}".format(x) for x in range(15)]
femaledata.columns = ["f{}".format(x) for x in range(15)]
maledata['label'] = 1
femaledata['label'] = 0

# 分割数据集，4:1
df = maledata.append(femaledata, ignore_index=True)
features = np.array(df[df.columns[0:15]], dtype='float32')
# features = torch.from_numpy(features)
labels = np.array(df[df.columns[15]], dtype='int32')
# labels = torch.from_numpy(labels).long()
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=2)
x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()

# 定义超参数
train_batch_size = 64
test_batch_size = 128
learning_rate = 0.01
n_epoch = 100
momentum = 0.5

# 获取迭代器
train_iter = Data.DataLoader(Data.TensorDataset(x_train, y_train), train_batch_size, shuffle=True)
test_iter = Data.DataLoader(Data.TensorDataset(x_test, y_test), test_batch_size, shuffle=True)

# n_input, n_output, n_hidden1, n_hidden2 = 15, 2, 256, 64

# net = nn.Sequential(
#     nn.Linear(n_input, n_hidden1), # 输入层与第一隐藏层结点数设置，全连接
#     nn.ReLU(),
#     nn.Linear(n_hidden1, n_hidden2),
#     nn.ReLU(),
#     nn.Linear(n_hidden2, n_output)
# ).to(device)

# 搭建网络
class Net(nn.Module):
    def __init__(self, in_dim, n_hidden1, n_hidden2, out_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden1), nn.BatchNorm1d(n_hidden1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden1, n_hidden2), nn.BatchNorm1d(n_hidden2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden2, out_dim))

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x



# 实例化网络
model = Net(15, 256, 64, 2)
model.to(device)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train():
    losses = []
    acces = []
    eval_losses = []
    eval_acces = []

    for epoch in range(n_epoch):
        """
        开始训练
        """
        train_loss = 0
        train_acc = 0
        model.train()
        # 动态修改学习率
        if epoch % 5 == 0:
            optimizer.param_groups[0]['lr'] *= 0.1
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            X = X.view(X.size(0), -1)
            # 前向传播
            y_hat = model(X)
            loss = criterion(y_hat, y)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 记录误差
            train_loss += loss.item()
            # 计算分类的准确率
            _, pred = y_hat.max(1)
            n_correct = (pred == y).sum().item()
            acc = n_correct / X.shape[0]
            train_acc += acc

        losses.append(train_loss / len(train_iter))
        acces.append(train_acc / len(train_iter))

        """
        在测试集上检验效果
        """
        eval_loss = 0
        eval_acc = 0
        model.eval()
        for X, y in test_iter:
            X = X.to(device)
            y = y.to(device)
            X = X.view(X.size(0), -1)
            y_hat = model(X)
            loss = criterion(y_hat, y)
            eval_loss += loss.item()
            _, pred = y_hat.max(1)
            n_correct = (pred == y).sum().item()
            acc = n_correct / X.shape[0]
            eval_acc += acc

        eval_losses.append(eval_loss / len(test_iter))
        eval_acces.append(eval_acc / len(test_iter))

        print('epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'.format(epoch, train_loss/len(train_iter),
                                                                                                             train_acc / len(train_iter),
                                                                                                             eval_loss / len(test_iter),
                                                                                                             eval_acc / len(test_iter)))

if __name__ == '__main__':
    """
    部分结果
    epoch: 33, Train Loss: 0.2670, Train Acc: 0.8881, Test Loss: 0.2684, Test Acc: 0.8851
    epoch: 34, Train Loss: 0.2653, Train Acc: 0.8882, Test Loss: 0.2691, Test Acc: 0.8840
    epoch: 35, Train Loss: 0.2647, Train Acc: 0.8879, Test Loss: 0.2661, Test Acc: 0.8855
    epoch: 36, Train Loss: 0.2660, Train Acc: 0.8873, Test Loss: 0.2677, Test Acc: 0.8839
    epoch: 37, Train Loss: 0.2649, Train Acc: 0.8881, Test Loss: 0.2663, Test Acc: 0.8857
    """
    train()

