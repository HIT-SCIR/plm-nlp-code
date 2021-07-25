# Defined in Section 5.3.1.2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from utils import BOS_TOKEN, EOS_TOKEN
from utils import load_reuters, save_pretrained, get_loader, init_weights

class NGramDataset(Dataset):
    def __init__(self, corpus, vocab, context_size=2):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        for sentence in tqdm(corpus, desc="Dataset Construction"):
            # 插入句首句尾符号
            sentence = [self.bos] + sentence + [self.eos]
            if len(sentence) < context_size:
                continue
            for i in range(context_size, len(sentence)):
                # 模型输入：长为context_size的上文
                context = sentence[i-context_size:i]
                # 模型输出：当前词
                target = sentence[i]
                self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        # 从独立样本集合中构建batch输入输出
        inputs = torch.tensor([ex[0] for ex in examples], dtype=torch.long)
        targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        return (inputs, targets)

class FeedForwardNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim):
        super(FeedForwardNNLM, self).__init__()
        # 词嵌入层
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 线性变换：词嵌入层->隐含层
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        # 线性变换：隐含层->输出层
        self.linear2 = nn.Linear(hidden_dim, vocab_size)
        # 使用ReLU激活函数
        self.activate = F.relu
        init_weights(self)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((inputs.shape[0], -1))
        hidden = self.activate(self.linear1(embeds))
        output = self.linear2(hidden)
        # 根据输出层（logits）计算概率分布并取对数，以便于计算对数似然
        # 这里采用PyTorch库的log_softmax实现
        log_probs = F.log_softmax(output, dim=1)
        return log_probs

embedding_dim = 64
context_size = 2
hidden_dim = 128
batch_size = 1024
num_epoch = 10

# 读取文本数据，构建FFNNLM训练数据集（n-grams）
corpus, vocab = load_reuters()
dataset = NGramDataset(corpus, vocab, context_size)
data_loader = get_loader(dataset, batch_size)

# 负对数似然损失函数
nll_loss = nn.NLLLoss()
# 构建FFNNLM，并加载至device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FeedForwardNNLM(len(vocab), embedding_dim, context_size, hidden_dim)
model.to(device)
# 使用Adam优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
total_losses = []
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(data_loader, desc=f"Training Epoch {epoch}"):
        inputs, targets = [x.to(device) for x in batch]
        optimizer.zero_grad()
        log_probs = model(inputs)
        loss = nll_loss(log_probs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss: {total_loss:.2f}")
    total_losses.append(total_loss)

# 保存词向量（model.embeddings）
save_pretrained(vocab, model.embeddings.weight.data, "ffnnlm.vec")

