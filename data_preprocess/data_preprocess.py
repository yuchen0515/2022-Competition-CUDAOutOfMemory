# coding: utf-8

import numpy as np
import pandas as pd
import re
import os
import json
import torch
import nltk
import collections

from nltk import ngrams
from tqdm import tqdm
from transformers import BertTokenizerFast
from ckiptagger import data_utils, WS

from nlp_fluency.models import NgramsLanguageModel


print("Convert Json to CSV...")
# Convert Json to CSV
with open('./data_preprocess/train_all.json', encoding='utf-8') as f:
    data = json.load(f)

ground_truth_list = []
sentence_list = []
phoneme_sequence_list = []
id_list = []

for d in tqdm(data):
    ground_truth = d['ground_truth_sentence']
    data_id = d['id']
    for sentence, phoneme in zip(d['sentence_list'], d['phoneme_sequence_list']):
        ground_truth_list.append(ground_truth)
        sentence_list.append(sentence)
        phoneme_sequence_list.append(phoneme)
        id_list.append(d['id'])

data_dict = {'ground_truth_sentence': ground_truth_list,
             'sentence_list': sentence_list,
             'phoneme_sequence_list': phoneme_sequence_list,
             'id': id_list}

output_df = pd.DataFrame(data=data_dict)
output_df.to_csv('./data_preprocess/train_all.csv', index=False)
print("Done")

print("Preprocessing...")
data = pd.read_csv('./data_preprocess/train_all.csv')
# Check Alphabets/numerals/Chinese numerals
re_filter_eng = r'[a-zA-Z]'
re_filter_numbers = r'\d+'
re_filter_chinese_numbers = r'[壹貳參肆伍陸柒捌玖拾仟]'


# # 分割音素

# In[12]:


def phoneme_split(phoneme):
    split_phoneme = re.split(r'(\d\s)', phoneme)
    for idx in range(int((len(split_phoneme) - 1) / 2)):
        split_phoneme[idx:idx+2] = [''.join(split_phoneme[idx:idx+2])]
    for idx in range(len(split_phoneme)):
        split_phoneme[idx] = split_phoneme[idx].replace(' ', '')
    for idx in range(len(split_phoneme) - 1, -1, -1):
        if split_phoneme[idx] == 's6' or split_phoneme[idx] == 'ts6':
            split_phoneme[idx:idx+2] = [''.join(split_phoneme[idx:idx+2])]

    return split_phoneme


# In[14]:


data['split_phoneme'] = data['phoneme_sequence_list'].apply(
    lambda x: phoneme_split(x))


# # 混淆字處理

# In[16]:


# 大寫數字處理
# 會個別處理，不會統一換成小寫數字
def num_convert(text, phoneme=None):
    # 壹貳柒捌玖仟 => 直接轉換
    converted_text = re.sub('壹', '一', text)
    converted_text = re.sub('貳', '二', converted_text)
    converted_text = re.sub('柒', '七', converted_text)
    converted_text = re.sub('捌', '八', converted_text)
    converted_text = re.sub('玖', '九', converted_text)
    converted_text = re.sub('仟', '千', converted_text)
    # 參肆伍陸拾 => 同音異義 / 異音異義的詞太多，暫不處理
    return converted_text

# 英文字處理
# 目前暫不轉換，直接刪除


def english_convert(text):
    converted_text = re.sub(r'[a-zA-Z]', '', text)
    return converted_text

# 特殊字處理
# 處理一些辨識正確，但output不符合主辦方規定之文字
# 目前只有 '5g' => '五居'


def special_word_convert(text):
    # '5g' => '五居'
    converted_text = text.replace('5g', '五居')

    return converted_text


# In[17]:


data['converted_sentence'] = data['sentence_list'].apply(
    lambda x: special_word_convert(x))
data['converted_sentence'] = data['converted_sentence'].apply(
    lambda x: english_convert(x))
data['converted_sentence'] = data['converted_sentence'].apply(
    lambda x: num_convert(x))
data['converted_sentence'] = data['converted_sentence'].apply(
    lambda x: x.replace(' ', ''))


# In[18]:


data['ground_truth_length'] = data['ground_truth_sentence'].apply(
    lambda x: len(x))
data['phoneme_length'] = data['split_phoneme'].apply(lambda x: len(x))
data['converted_sentence_length'] = data['converted_sentence'].apply(
    lambda x: len(x))

data.to_csv('./data_preprocess/preprocessed_train_all.csv', index=False)

print("Done")

# Build N-Gram Language Model
print("building N-Gram LM...")

data = pd.read_csv('./data_preprocess/preprocessed_train_all.csv')

if not os.path.exists("./data"):
    print("ckiptagger model not exists, Downloading...")
    data_utils.download_data_url("./")
    print("Downloaded")
ws = WS('./data')

ground_truth = list(set(data['ground_truth_sentence'].values))
ground_truth = ws(ground_truth)

text = nltk.text.TextCollection(ground_truth)

uni_gram = ngrams(text, 1)
bi_gram = ngrams(text, 2)
tri_gram = ngrams(text, 3)

uni_gram_counter = collections.Counter(uni_gram)
bi_gram_counter = collections.Counter(bi_gram)
tri_gram_counter = collections.Counter(tri_gram)

uni_gram_dict = dict((key[0], value)
                     for key, value in uni_gram_counter.items())
bi_gram_dict = dict((' '.join(key[0:]), value)
                    for key, value in bi_gram_counter.items())
tri_gram_dict = dict((' '.join(key[0:]), value)
                     for key, value in tri_gram_counter.items())

if not os.path.exists('./ngram_lm'):
    os.mkdir('./ngram_lm')

with open('./ngram_lm/unigram.json', 'w') as fp:
    json.dump(uni_gram_dict, fp)

with open('./ngram_lm/bigram.json', 'w') as fp:
    json.dump(bi_gram_dict, fp)

with open('./ngram_lm/trigram.json', 'w') as fp:
    json.dump(tri_gram_dict, fp)

print('Done')

# Build NLP Fluency N-Gram LM
ngram_lm = NgramsLanguageModel(ngram=3, sentence_length=200)
ngram_lm.train(ground_truth)

ngram_lm.save("./ngram_lm/trigram")
