# plm-nlp-book

本仓库用于存放《[自然语言处理：基于预训练模型的方法](https://item.jd.com/13344628.html)》（作者：车万翔、郭江、崔一鸣）一书各章节的示例代码。


### 本书代码测试环境
* Python: 3.8.5
* PyTorch: 1.8.0
* Transformers: 4.9.0
* NLTK: 3.5
* LTP: 4.0

### 勘误
* 书中3.4.3节`convert_t2s.py`：
```python
f_in = open(sys.argv[0], "r")
```
修正为
```python
f_in = open(sys.argv[1], "r")
```

* 书中3.4.3节`wikidata_cleaning.py`：
```python
f_in = open(sys.argv[0], 'r')
```
修正为
```python
f_in = open(sys.argv[1], 'r')
```
此外，为了兼容Python 3.7以上版本，将`remove_control_chars`函数修改为：
```python
def remove_control_chars(in_str):
    control_chars = ''.join(map(chr, list(range(0, 32)) + list(range(127, 160))))
    control_chars = re.compile('[%s]' % re.escape(control_chars))
    return control_chars.sub('', in_str)
```

* 书中4.6.1节`Vocab`类的`__init__`与`build`方法有误，修正为：
```python
class Vocab:
    def __init__(self, tokens=None):
        self.idx_to_token = list()
        self.token_to_idx = dict()

        if tokens is not None:
            if "<unk>" not in tokens:
                tokens = tokens + ["<unk>"]
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
            self.unk = self.token_to_idx['<unk>']

    @classmethod
    def build(cls, text, min_freq=1, reserved_tokens=None):
        token_freqs = defaultdict(int)
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token, freq in token_freqs.items() \
                        if freq >= min_freq and token != "<unk>"]
        return cls(uniq_tokens)
```

* 书中4.6.5节使用的`MLP`模型类是基于`EmbeddingBag`的`MLP`实现，与4.6.3节的`MLP`实现有所区别，具体如下：
```python
class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(MLP, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.activate = F.relu
        self.linear2 = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs, offsets):
        embedding = self.embedding(inputs, offsets)
        hidden = self.activate(self.linear1(embedding))
        outputs = self.linear2(hidden)
        log_probs = F.log_softmax(outputs, dim=1)
        return log_probs
```

* 书中6.2.3节`ELMoLstmEncoder`类的`forward`函数实现有误，修正为：
```python
def forward(self, inputs, lengths):
    batch_size, seq_len, input_dim = inputs.shape
    rev_idx = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    for i in range(lengths.shape[0]):
        rev_idx[i,:lengths[i]] = torch.arange(lengths[i]-1, -1, -1)
    rev_idx = rev_idx.unsqueeze(2).expand_as(inputs)
    rev_idx = rev_idx.to(inputs.device)
    rev_inputs = inputs.gather(1, rev_idx)

    forward_inputs, backward_inputs = inputs, rev_inputs
    stacked_forward_states, stacked_backward_states = [], []

    for layer_index in range(self.num_layers):
        # Transfer `lengths` to CPU to be compatible with latest PyTorch versions.
        packed_forward_inputs = pack_padded_sequence(
            forward_inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_backward_inputs = pack_padded_sequence(
            backward_inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # forward
        forward_layer = self.forward_layers[layer_index]
        packed_forward, _ = forward_layer(packed_forward_inputs)
        forward = pad_packed_sequence(packed_forward, batch_first=True)[0]
        forward = self.forward_projections[layer_index](forward)
        stacked_forward_states.append(forward)

        # backward
        backward_layer = self.backward_layers[layer_index]
        packed_backward, _ = backward_layer(packed_backward_inputs)
        backward = pad_packed_sequence(packed_backward, batch_first=True)[0]
        backward = self.backward_projections[layer_index](backward)
        # convert back to original sequence order using rev_idx
        stacked_backward_states.append(backward.gather(1, rev_idx))

        forward_inputs, backward_inputs = forward, backward

    # stacked_forward_states: [batch_size, seq_len, projection_dim] * num_layers
    # stacked_backward_states: [batch_size, seq_len, projection_dim] * num_layers
    return stacked_forward_states, stacked_backward_states
```

* 书中7.4.3节（199页）"句对文本分类"→"代码实现"中的`tokenize()`函数存在问题，请按如下进行修正。
```python
def tokenize(examples):
    return tokenizer(examples['hypothesis'], examples['premise'], truncation=True, padding='max_length')
```
修正为
```python
def tokenize(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length')
```