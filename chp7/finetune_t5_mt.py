import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

# 加载并处理数据集
model_name = "google/mt5-small"     # 此处也可以选用更大的模型版本
iwslt_data = load_dataset("iwslt2017", "iwslt2017-zh-en")
tokenizer = AutoTokenizer.from_pretrained(model_name)

source_lang = "zh"
target_lang = "en"
prefix = "translate Chinese to English: "

def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

tokenized_data = iwslt_data.map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)

# 定义评价方法
metric = evaluate.load("sacrebleu")
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# 定义模型、训练超参
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

training_args = Seq2SeqTrainingArguments(
    output_dir="iwslt_zh_en_model",     # 输出路径，存放检查点和其他输出文件
    evaluation_strategy="epoch",        # 定义每轮结束后进行评价
    learning_rate=2e-5,                 # 定义初始学习率
    per_device_train_batch_size=64,     # 定义训练批次大小
    per_device_eval_batch_size=64,      # 定义测试批次大小
    weight_decay=0.01,                  # 定义优化器权重衰减系数
    save_total_limit=3,                 # 定义最多保存多少个检查点  
    num_train_epochs=2,                 # 定义训练轮数
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 开始训练！
trainer.train()