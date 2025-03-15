# Defined in Section 7.4.4.2

import numpy as np
from datasets import load_dataset, load_metric
from transformers import BertTokenizerFast, BertForQuestionAnswering, TrainingArguments, Trainer, default_data_collator

# 加载训练数据、分词器、预训练模型以及评价方法
dataset = load_dataset('squad')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
model = BertForQuestionAnswering.from_pretrained('bert-base-cased', return_dict=True)
metric = load_metric('squad')

# 准备训练数据并转换为feature
def prepare_train_features(examples):
    tokenized_examples = tokenizer(
        examples["question"],           # 问题文本
        examples["context"],            # 篇章文本
        truncation="only_second",       # 截断只发生在第二部分，即篇章
        max_length=384,                 # 设定最大长度为384
        stride=128,                     # 设定篇章切片步长为128
        return_overflowing_tokens=True, # 返回超出最大长度的标记，将篇章切成多片
        return_offsets_mapping=True,    # 返回偏置信息，用于对齐答案位置
        padding="max_length",           # 按最大长度进行补齐
    )

    # 如果篇章很长，则可能会被切成多个小篇章，需要通过以下函数建立feature到example的映射关系
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # 建立token到原文的字符级映射关系，用于确定答案的开始和结束位置
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # 获取开始和结束位置
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # 获取输入序列的input_ids以及[CLS]标记的位置（在BERT中为第0位）
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # 获取哪些部分是问题，哪些部分是篇章
        sequence_ids = tokenized_examples.sequence_ids(i)

        # 获取答案在文本中的字符级开始和结束位置
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        # 获取在当前切片中的开始和结束位置
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # 检测答案是否超出当前切片的范围
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            # 超出范围时，答案的开始和结束位置均设置为[CLS]标记的位置
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # 将token_start_index和token_end_index移至答案的两端
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

# 通过函数prepare_train_features，建立分词后的训练集
tokenized_datasets = dataset.map(prepare_train_features, batched=True, remove_columns=dataset["train"].column_names)

# 定义训练参数TrainingArguments，默认使用AdamW优化器
args = TrainingArguments(
    "ft-squad",                         # 输出路径，存放检查点和其他输出文件
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
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=default_data_collator,
    tokenizer=tokenizer,
)

# 开始训练！（主流GPU上耗时约几小时）
trainer.train()
