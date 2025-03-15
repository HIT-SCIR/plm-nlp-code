# 第9章：大语言模型的适配
### 9.6.1 中文词表扩充

```
python merge_tokenizers.py --llama_tokenizer_dir original_llama_tokenizer_dir --chinese_sp_model_file zh_vocab.model
```

### 9.7.1 知识蒸馏

```
python textbrewer_example.py
```

### 9.7.2 模型裁剪

```
python textpruner_example.py
```
