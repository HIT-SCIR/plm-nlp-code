import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, 

# 加载并处理数据集
model_name = "gpt2"
wikitext_data = load_dataset("wikitext", "wikitext-2-v1")
tokenizer = AutoTokenizer.from_pretrained(model_name)
block_size = 128

def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["text"]])

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_wikitext = wikitext_data.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=wikitext_data["train"].column_names,
)
lm_dataset = tokenized_wikitext.map(group_texts, batched=True, num_proc=4)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 定义模型、训练超参
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

training_args = TrainingArguments(
    output_dir="gpt2_wikitext_model",   # 输出路径，存放检查点和其他输出文件
    evaluation_strategy="epoch",        # 定义每轮结束后进行评价
    learning_rate=2e-5,                 # 定义初始学习率
    per_device_train_batch_size=32,     # 定义训练批次大小
    per_device_eval_batch_size=32,      # 定义测试批次大小
    weight_decay=0.01,                  # 定义优化器权重衰减系数
    num_train_epochs=2,                 # 定义训练轮数
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)

# 开始训练！
trainer.train()
