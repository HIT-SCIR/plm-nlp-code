# 第7章：预训练语言模型
## 7.5 预训练模型的任务微调：NLU类任务
### 7.5.1 单句文本分类
```
python finetune_bert_ssc.py
```

### 7.5.2 句对文本分类
```
python finetune_bert_spc.py
```

### 7.5.3 阅读理解
```
python finetune_bert_mrc.py
```

### 7.5.4 序列标注（命名实体识别）
```
python finetune_bert_ner.py
```

## 7.6 预训练模型的任务微调：NLG类任务
### 7.6.1 文本生成
```
python finetune_gpt2_tg.py
```

### 7.6.2 机器翻译
```
python finetune_t5_mt.py
```