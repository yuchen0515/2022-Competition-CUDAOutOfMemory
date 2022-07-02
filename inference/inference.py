import numpy as np
import pandas as pd
import json
import random
import time
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, BertTokenizerFast

from ckiptagger import WS, data_utils

from dataset_inference import *
from config import Config
from word_converter import *
from model import *
from perplexity import *
from search_method import *



def inference(model, tokenizer, device, sentence_list, Config):
    '''
        inference input sentences to corrected sentences
        
        params:
            param1 model: Trained Transformer model
            param2 tokenizer: a BERT tokenizer for sentence tokenize
            param3 device: a string which determines the graphic card model/tensor will be used
            param4 sentence_list: a list of string for sentence inference
            param5 Config: a Config class for model setting
        
        return:
            predicted_sentences: a list of string corrected by model
    '''
    model.to(device)
    model.eval()
    
    sentence_list = word_converter(sentence_list)

    inference_dataset = GecDatasetInference(sentence_list, tokenizer, Config)
    inference_dataloader = DataLoader(dataset=inference_dataset, batch_size=inference_dataset.len, collate_fn=collate_fn_inference)
    
    corrected_sentence_list = []
    for batch in inference_dataloader:
        corrected_sentence_list = greedy_search(model, tokenizer, device, 
                                                batch, inference_dataloader.batch_size)
              
    
    return sentence_list, corrected_sentence_list




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None, help='Your input json file path')
    parser.add_argument('--model_path', type=str, help='Your inference model path')
    parser.add_argument('--tokenizer_path', type=str, default='bert-base-chinese', help='Your pre-trained tokenizer model path')
    parser.add_argument('--device', type=str, default='cuda:0', help='Graphic card(or cpu) which used in model inference')
    parser.add_argument('--ws_path', type=str, default='../data', help='pre-trained(?) model of ckip word segment model')
    parser.add_argument('--ngram_dict_dir', type=str, default=None, help='directory of ngrams dictionary')
    
    args = parser.parse_args()
    
    input_path = args.input_path
    model_path = args.model_path
    tokenizer_path = args.tokenizer_path
    device = args.device
    if device != 'cpu':
        device = device if torch.cuda.is_available() else 'cpu'
    
    ws_path = args.ws_path
    ngram_dict_dir = args.ngram_dict_dir
    
    model = torch.load(model_path, map_location=device)
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    
    with open(ngram_dict_dir + '/' +'unigram.json') as dict:
        uni_gram_dict = json.load(dict)
    with open(ngram_dict_dir + '/' +'bigram.json') as dict:
        bi_gram_dict = json.load(dict)
    with open(ngram_dict_dir + '/' +'trigram.json') as dict:
        tri_gram_dict = json.load(dict)
    
    if not os.path.exists(ws_path):
        print("ckiptagger model not exists, Downloading...")
        data_utils.download_data_url("../")
        ws_path = '../data'
        print("Downloaded")
    ws = WS(ws_path)
    
    start = time.time()
    if input_path == None:
        #some sample
        sentence_list = ['信用卡', '信用卡版', '信用卡 嗎', '信用卡 噢', '信用卡款']
    else:
        with open(input_path, encoding='utf-8') as f:
            data = json.load(f)
            sentence_list = data['sentence_list']
    

    sentence_list, corrected_sentence_list = inference(model, tokenizer, device, sentence_list, Config)
    
    corrected_perpelxity_list = trigram_perplexity(corrected_sentence_list, 
                                                   ws, uni_gram_dict, bi_gram_dict, tri_gram_dict)
    
    ranking = np.argsort(corrected_perpelxity_list)
    best_sentence = corrected_sentence_list[ranking[0]]
    
    end = time.time()
    
    print('Source sentences:', sentence_list)
    print(f'The best perplexity sentnece:{best_sentence}({corrected_perpelxity_list[ranking[0]]})')
    print('inference time:', end-start)