import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from transformers import XLMRobertaForSequenceClassification,XLMRobertaTokenizer
from textpruner import summary, TransformerPruner, TransformerPruningConfig
import sys, os

sys.path.insert(0, os.path.abspath('..'))

from classification_utils.dataloader_script import eval_dataset, dataloader, eval_langs, batch_size
from classification_utils.predict_function import predict

model_path = 'ziqingyang/XLMRobertaBaseForPAWSX-en'
model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)

print("Before pruning:")
print(summary(model))

transformer_pruning_config = TransformerPruningConfig(
    target_ffn_size=2048, target_num_of_heads=8, 
    pruning_method='iterative',n_iters=4)
pruner = TransformerPruner(model,transformer_pruning_config=transformer_pruning_config)   
pruner.prune(dataloader=dataloader, save_model=True)

# save the tokenizer to the same place
tokenizer.save_pretrained(pruner.save_dir)

print("After pruning:")
print(summary(model))

for i in range(12):
    print ((model.base_model.encoder.layer[i].intermediate.dense.weight.shape,
            model.base_model.encoder.layer[i].intermediate.dense.bias.shape,
            model.base_model.encoder.layer[i].attention.self.key.weight.shape))


print("Measure performance")
device= model.device
eval_datasets = [eval_dataset.lang_datasets[lang] for lang in eval_langs]

predict(model, eval_datasets, eval_langs, device, batch_size)
