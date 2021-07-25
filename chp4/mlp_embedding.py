import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(MLP, self).__init__()
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 线性变换：词嵌入层->隐含层
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        # 使用ReLU激活函数
        self.activate = F.relu
        # 线性变换：激活层->输出层
        self.linear2 = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        # 将序列中多个embedding进行聚合（此处是求平均值）
        embedding = embeddings.mean(dim=1)
        hidden = self.activate(self.linear1(embedding))
        outputs = self.linear2(hidden)
        # 获得每个序列属于某一类别概率的对数值
        probs = F.log_softmax(outputs, dim=1)
        return probs

mlp = MLP(vocab_size=8, embedding_dim=3, hidden_dim=5, num_class=2)
# 输入为两个长度为4的整数序列
inputs = torch.tensor([[0, 1, 2, 1], [4, 6, 6, 7]], dtype=torch.long)
outputs = mlp(inputs)
print(outputs)