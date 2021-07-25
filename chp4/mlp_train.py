import torch
from torch import nn, optim
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activate = F.relu
        self.linear2 = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs):
        hidden = self.linear1(inputs)
        activation = self.activate(hidden)
        outputs = self.linear2(activation)
        # 获得每个输入属于某一类别的概率（Softmax），然后再取对数
        # 取对数的目的是避免计算Softmax时可能产生的数值溢出问题
        log_probs = F.log_softmax(outputs, dim=1)
        return log_probs

# 异或问题的4个输入
x_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
# 每个输入对应的输出类别
y_train = torch.tensor([0, 1, 1, 0])

# 创建多层感知器模型，输入层大小为2，隐含层大小为5，输出层大小为2（即有两个类别）
model = MLP(input_dim=2, hidden_dim=5, num_class=2)

criterion = nn.NLLLoss() # 当使用log_softmax输出时，需要调用负对数似然损失（Negative Log Likelihood，NLL）
optimizer = optim.SGD(model.parameters(), lr=0.05) # 使用梯度下降参数优化方法，学习率设置为0.05

for epoch in range(100):
    y_pred = model(x_train) # 调用模型，预测输出结果
    loss = criterion(y_pred, y_train) # 通过对比预测结果与正确的结果，计算损失
    optimizer.zero_grad() # 在调用反向传播算法之前，将优化器的梯度值置为零，否则每次循环的梯度将进行累加
    loss.backward() # 通过反向传播计算参数的梯度
    optimizer.step() # 在优化器中更新参数，不同优化器更新的方法不同，但是调用方式相同

print("Parameters:")
for name, param in model.named_parameters():
    print (name, param.data)

y_pred = model(x_train)
print("Predicted results:", y_pred.argmax(axis=1))
