# Defined in Section 5.2.3.2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from utils import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from utils import load_reuters, save_pretrained, get_loader, init_weights

class SkipGramDataset(Dataset):
    def __init__(self, corpus, vocab, context_size=2):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        for sentence in tqdm(corpus, desc="Dataset Construction"):
            sentence = [self.bos] + sentence + [self.eos]
            for i in range(1, len(sentence)-1):
                # 模型输入：当前词
                w = sentence[i]
                # 模型输出：一定窗口大小内的上下文
                left_context_index = max(0, i - context_size)
                right_context_index = min(len(sentence), i + context_size)
                context = sentence[left_context_index:i] + sentence[i+1:right_context_index+1]
                self.data.extend([(w, c) for c in context])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        inputs = torch.tensor([ex[0] for ex in examples])
        targets = torch.tensor([ex[1] for ex in examples])
        return (inputs, targets)

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)
        init_weights(self)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        output = self.output(embeds)
        log_probs = F.log_softmax(output, dim=1)
        return log_probs

embedding_dim = 64
context_size = 2
hidden_dim = 128
batch_size = 1024
num_epoch = 10

# 读取文本数据，构建Skip-gram模型训练数据集
corpus, vocab = load_reuters()
dataset = SkipGramDataset(corpus, vocab, context_size=context_size)
data_loader = get_loader(dataset, batch_size)

nll_loss = nn.NLLLoss()
# 构建Skip-gram模型，并加载至device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SkipGramModel(len(vocab), embedding_dim)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
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

# 保存词向量（model.embeddings）
save_pretrained(vocab, model.embeddings.weight.data, "skipgram.vec")

