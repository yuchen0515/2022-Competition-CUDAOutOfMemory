import numpy as np
import pandas as pd
import math
import random
import re
import os
from tqdm import tqdm
from ckiptagger import WS, data_utils
import json

from nlp_fluency.models import NgramsLanguageModel

# torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.modules import transformer

# transformers
from transformers import BertModel, BertTokenizerFast
from evaluate import load

from model.model import *
from model.data import *
from model.evaluation import *
from model.config import Config

if not os.path.exists(Config.model_dir):
    os.mkdir(Config.model_dir)


def train(model, tokenizer, device, criterion, optimizer,
          train_dataloader, dev_data,
          ws, uni_gram_dict, bi_gram_dict, tri_gram_dict,
          nlp_fluency_lm,
          model_dir,
          epoch=5, max_norm=1.0):

    model.to(device)

    best_cer = 1e9
    best_model_path = ''
    for e in range(epoch):
        model.train()

        with tqdm(total=len(train_dataloader)) as pbar:
            pbar.set_description('Train Epoch %d' % (e+1))
            # train part
            model.train()
            training_loss = 0
            for idx, batch in enumerate(train_dataloader):
                src_input_ids = batch[0].to(device)
                src_key_padding_mask = batch[1].to(device)
                trg_input_ids = batch[3].to(device)
                trg_key_padding_mask = batch[4].to(device)

                src_mask = generate_mask(src_input_ids.shape[1]).to(device)

                trg_inputs = trg_input_ids[:, :-1]
                trg_outputs = trg_input_ids[:, 1:]
                trg_mask = generate_mask(trg_inputs.shape[1]).to(device)

                memory_mask = src_key_padding_mask.clone()
                trg_key_padding_mask = trg_key_padding_mask[:, :-1]

                output = model(src_input_ids, trg_inputs,
                               src_key_padding_mask, trg_key_padding_mask,
                               src_mask, trg_mask,
                               memory_mask)

                loss = criterion(output.contiguous().view(-1, output.size(-1)),
                                 trg_outputs.contiguous().view(-1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)
                optimizer.step()
                model.zero_grad()

                training_loss += loss.item()
                pbar.set_postfix(average_loss=training_loss/(idx+1))
                pbar.update(1)
        # eval model with dev data
        curr_model_path = f'{model_dir}/epoch_{e+1}.bin'
        torch.save(model.state_dict(), curr_model_path)

        curr_cer = evaluation(model, tokenizer, device, dev_data,
                              ws, uni_gram_dict, bi_gram_dict, tri_gram_dict,
                              nlp_fluency_lm)
        if curr_cer < best_cer:
            best_cer = curr_cer
            best_model_path = curr_model_path

    print('Best Model Path:', best_model_path)
    return best_model_path


with open(Config.ngram_dict_dir + '/' + 'unigram.json') as d:
    uni_gram_dict = json.load(d)
with open(Config.ngram_dict_dir + '/' + 'bigram.json') as d:
    bi_gram_dict = json.load(d)
with open(Config.ngram_dict_dir + '/' + 'trigram.json') as d:
    tri_gram_dict = json.load(d)

nlp_fluency_lm = NgramsLanguageModel.from_pretrained(
    Config.nlp_fluency_lm_path)

if not os.path.exists("./data"):
    data_utils.download_data_url("./")
ws = WS('./data')


train_data, dev_data = data_handler(data_path=f'./data_preprocess/preprocessed_train_all.csv',
                                    output_path_list=[f'{Config.model_dir}/train_data.csv',
                                                      f'{Config.model_dir}/dev_data.csv'],
                                    sample_or_split='split',
                                    sample_or_train_ratio=0.8,
                                    max_len=200)


curr_dir = f'{Config.model_dir}'
if not os.path.exists(curr_dir):
    os.mkdir(curr_dir)

torch.cuda.empty_cache()

device = Config.device
tokenizer = BertTokenizerFast.from_pretrained(Config.model_path)
model = GecTransformer(max_len=200,
                       num_of_vocab=tokenizer.vocab_size,
                       d_model=512,
                       nhead=8,
                       num_encoder_layers=6,
                       num_decoder_layers=6,
                       dim_feedforward=2048,
                       dropout=0.2,
                       activation="relu")
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=Config.lr)

train_dataset = GecDataset(train_data, tokenizer, Config)
train_dataloader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=Config.batch_size,
                              collate_fn=collate_fn)


best_model_path = train(model, tokenizer, Config.device, criterion, optimizer,
                        train_dataloader, dev_data,
                        ws, uni_gram_dict, bi_gram_dict, tri_gram_dict,
                        nlp_fluency_lm,
                        curr_dir,
                        epoch=6, max_norm=Config.max_norm)
