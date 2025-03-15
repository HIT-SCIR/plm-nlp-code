import os
import re
import sentencepiece as spm
import argparse
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model

import logging
logging.basicConfig(level=logging.INFO)

def load_model(model_file):
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(model_file)
    return sp_model

def find_english_tokens_and_punctuations(model_proto):
    en_words = {p.piece for p in model_proto.pieces if re.findall("[a-zA-Z]+", p.piece)}
    punct_ps = {p.piece for p in model_proto.pieces if not re.search(r'(\w|\d)+', p.piece) and len(p.piece.lstrip('▁')) > 1}
    return en_words, punct_ps

def merge_tokenizers(llama_model_proto, chinese_model_proto, en_words, punct_ps):
    llama_tokens_set = {p.piece for p in llama_model_proto.pieces}
    logging.info(f"Initial Llama tokenizer size: {len(llama_tokens_set)}")
    
    for p in chinese_model_proto.pieces:
        if p.piece not in llama_tokens_set and p.piece not in en_words and p.piece not in punct_ps:
            llama_model_proto.pieces.add(sp_pb2_model.ModelProto.SentencePiece(piece=p.piece, score=0))
            if len(llama_model_proto.pieces) == 32000:
                llama_model_proto.pieces.add(sp_pb2_model.ModelProto.SentencePiece(piece='<pad>', score=0))
                break

    logging.info(f"New model pieces: {len(llama_model_proto.pieces)}")

def save_merged_model(model_proto, output_sp_dir, output_hf_dir):
    os.makedirs(output_sp_dir, exist_ok=True)
    with open(os.path.join(output_sp_dir, 'chinese_llama.model'), 'wb') as f:
        f.write(model_proto.SerializeToString())

    tokenizer = LlamaTokenizer(vocab_file=os.path.join(output_sp_dir, 'chinese_llama.model'))
    tokenizer.save_pretrained(output_hf_dir)
    logging.info(f"Chinese-Llama tokenizer has been saved to {output_hf_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--llama_tokenizer_file', required=True)
    parser.add_argument('--chinese_sp_model_file', default='./chinese_sp.model')
    args = parser.parse_args()

    llama_sp_model = load_model(args.llama_tokenizer_file)
    chinese_sp_model = load_model(args.chinese_sp_model_file)

    llama_sp_mp = sp_pb2_model.ModelProto()
    llama_sp_mp.ParseFromString(llama_sp_model.serialized_model_proto())
    chinese_uni_sp_mp = sp_pb2_model.ModelProto()
    chinese_uni_sp_mp.ParseFromString(chinese_sp_model.serialized_model_proto())

    en_words, punct_ps = find_english_tokens_and_punctuations(chinese_uni_sp_mp)
    merge_tokenizers(llama_sp_mp, chinese_uni_sp_mp, en_words, punct_ps)

    output_sp_dir = 'merged_tokenizer_sp'
    output_hf_dir = 'merged_tokenizer_hf'
    save_merged_model(llama_sp_mp, output_sp_dir, output_hf_dir)