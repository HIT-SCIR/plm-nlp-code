# Defined in Section 7.4.5.2

import numpy as np
from datasets import load_dataset, load_metric
from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification

# 加载CoNLL-2003数据集、分词器
dataset = load_dataset('conll2003')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

# 将训练集转换为可训练的特征形式
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True,  is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # 将特殊符号的标签设置为-100，以便在计算损失函数时自动忽略
            if word_idx is None:
                label_ids.append(-100)
            # 把标签设置到每个词的第一个token上
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # 对于每个词的其他token也设置为当前标签
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True, load_from_cache_file=False)

# 获取标签列表，并加载预训练模型
label_list = dataset["train"].features["ner_tags"].feature.names
model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(label_list))

# 定义data_collator，并使用seqeval进行评价
data_collator = DataCollatorForTokenClassification(tokenizer)
metric = load_metric("seqeval")

# 定义评价指标
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # 移除需要忽略的下标（之前记为-100）
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# 定义训练参数TrainingArguments和Trainer
args = TrainingArguments(
    "ft-conll2003",                     # 输出路径，存放检查点和其他输出文件
    evaluation_strategy="epoch",        # 定义每轮结束后进行评价
    learning_rate=2e-5,                 # 定义初始学习率
    per_device_train_batch_size=16,     # 定义训练批次大小
    per_device_eval_batch_size=16,      # 定义测试批次大小
    num_train_epochs=3,                 # 定义训练轮数
)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 开始训练！（主流GPU上耗时约几分钟）
trainer.train()
