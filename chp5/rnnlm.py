# Defined in Section 5.1.3.3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from utils import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from utils import load_reuters, save_pretrained, get_loader, init_weights

class RnnlmDataset(Dataset):
    def __init__(self, corpus, vocab):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        self.pad = vocab[PAD_TOKEN]
        for sentence in tqdm(corpus, desc="Dataset Construction"):
            # 模型输入：BOS_TOKEN, w_1, w_2, ..., w_n
            input = [self.bos] + sentence
            # 模型输出：w_1, w_2, ..., w_n, EOS_TOKEN
            target = sentence + [self.eos]
            self.data.append((input, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        # 从独立样本集合中构建batch输入输出
        inputs = [torch.tensor(ex[0]) for ex in examples]
        targets = [torch.tensor(ex[1]) for ex in examples]
        # 对batch内的样本进行padding，使其具有相同长度
        inputs = pad_sequence(inputs, batch_first=True, padding_value=self.pad)
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad)
        return (inputs, targets)

class RNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNLM, self).__init__()
        # 词嵌入层
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 循环神经网络：这里使用LSTM
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # 输出层
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        # 计算每一时刻的隐含层表示
        hidden, _ = self.rnn(embeds)
        output = self.output(hidden)
        log_probs = F.log_softmax(output, dim=2)
        return log_probs

embedding_dim = 64
context_size = 2
hidden_dim = 128
batch_size = 1024
num_epoch = 10

# 读取文本数据，构建FFNNLM训练数据集（n-grams）
corpus, vocab = load_reuters()
dataset = RnnlmDataset(corpus, vocab)
data_loader = get_loader(dataset, batch_size)

# 负对数似然损失函数，忽略pad_token处的损失
nll_loss = nn.NLLLoss(ignore_index=dataset.pad)
# 构建RNNLM，并加载至device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNNLM(len(vocab), embedding_dim, hidden_dim)
model.to(device)
# 使用Adam优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(data_loader, desc=f"Training Epoch {epoch}"):
        inputs, targets = [x.to(device) for x in batch]
        optimizer.zero_grad()
        log_probs = model(inputs)
        loss = nll_loss(log_probs.view(-1, log_probs.shape[-1]), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss: {total_loss:.2f}")

save_pretrained(vocab, model.embeddings.weight.data, "rnnlm.vec")

