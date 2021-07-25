# Defined in Section 7.4.2.2

import numpy as np
from datasets import load_dataset, load_metric
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer

# 加载训练数据、分词器、预训练模型以及评价方法
dataset = load_dataset('glue', 'sst2')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-cased', return_dict=True)
metric = load_metric('glue', 'sst2')

# 对训练集进行分词
def tokenize(examples):
    return tokenizer(examples['sentence'], truncation=True, padding='max_length')
dataset = dataset.map(tokenize, batched=True)
encoded_dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)

# 将数据集格式化为torch.Tensor类型以训练PyTorch模型
columns = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
encoded_dataset.set_format(type='torch', columns=columns)

# 定义评价指标
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=np.argmax(predictions, axis=1), references=labels)

# 定义训练参数TrainingArguments，默认使用AdamW优化器
args = TrainingArguments(
    "ft-sst2",                          # 输出路径，存放检查点和其他输出文件
    evaluation_strategy="epoch",        # 定义每轮结束后进行评价
    learning_rate=2e-5,                 # 定义初始学习率
    per_device_train_batch_size=16,     # 定义训练批次大小
    per_device_eval_batch_size=16,      # 定义测试批次大小
    num_train_epochs=2,                 # 定义训练轮数
)

# 定义Trainer，指定模型和训练参数，输入训练集、验证集、分词器以及评价函数
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 开始训练！（主流GPU上耗时约几小时）
trainer.train()
