# Defined in Section 4.7.2

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from collections import defaultdict
from vocab import Vocab
from utils import load_treebank

#tqdm是一个Python模块，能以进度条的方式显式迭代的进度
from tqdm.auto import tqdm

WEIGHT_INIT_RANGE = 0.1

class LstmDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

def collate_fn(examples):
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = [torch.tensor(ex[1]) for ex in examples]
    inputs = pad_sequence(inputs, batch_first=True, padding_value=vocab["<pad>"])
    targets = pad_sequence(targets, batch_first=True, padding_value=vocab["<pad>"])
    return inputs, lengths, targets, inputs != vocab["<pad>"]


def init_weights(model):
    for param in model.parameters():
        torch.nn.init.uniform_(param, a=-WEIGHT_INIT_RANGE, b=WEIGHT_INIT_RANGE)

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(LSTM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, num_class)
        init_weights(self)

    def forward(self, inputs, lengths):
        embeddings = self.embeddings(inputs)
        x_pack = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        hidden, (hn, cn) = self.lstm(x_pack)
        hidden, _ = pad_packed_sequence(hidden, batch_first=True)
        outputs = self.output(hidden)
        log_probs = F.log_softmax(outputs, dim=-1)
        return log_probs

embedding_dim = 128
hidden_dim = 256
batch_size = 32
num_epoch = 5

#加载数据
train_data, test_data, vocab, pos_vocab = load_treebank()
train_dataset = LstmDataset(train_data)
test_dataset = LstmDataset(test_data)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

num_class = len(pos_vocab)

#加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTM(len(vocab), embedding_dim, hidden_dim, num_class)
model.to(device) #将模型加载到GPU中（如果已经正确安装）

#训练过程
nll_loss = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) #使用Adam优化器

model.train()
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
        inputs, lengths, targets, mask = [x.to(device) for x in batch]
        log_probs = model(inputs, lengths)
        loss = nll_loss(log_probs[mask], targets[mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss: {total_loss:.2f}")

#测试过程
acc = 0
total = 0
for batch in tqdm(test_data_loader, desc=f"Testing"):
    inputs, lengths, targets, mask = [x.to(device) for x in batch]
    with torch.no_grad():
        output = model(inputs, lengths)
        acc += (output.argmax(dim=-1) == targets)[mask].sum().item()
        total += mask.sum().item()

#输出在测试集上的准确率
print(f"Acc: {acc / total:.2f}")
