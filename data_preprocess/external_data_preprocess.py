import os
import requests
import tarfile
import json
import numpy as np
import pandas as pd
import random

from opencc import OpenCC
from tqdm import tqdm
from transformers import BertTokenizerFast, BertTokenizer

if not os.path.exists('./data_preprocess/clmad.tgz'):
    URL = 'https://www.openslr.org/resources/55/train.tgz'
    response = requests.get(URL)
    with open('./data_preprocess/clmad.tgz', 'wb') as f:
        f.write(response.content)

if not os.path.exists('./data_preprocess/clmad'):
    os.mkdir('./data_preprocess/clmad')
    file = tarfile.open('./data_preprocess/clmad.tgz')
    file.extractall('./data_preprocess/clmad')

cc = OpenCC('s2twp')
file_prefix_list = ['stock', 'finance']

for file_prefix in file_prefix_list:
    s2twp_corpus = []
    with open('./data_preprocess/clmad/' + file_prefix + '.train', 'r') as f:
        counter = 0
        for line in tqdm(f.readlines()):
            s2twp_corpus.append(cc.convert(line.strip('\n')))

    with open('./data_preprocess/' + file_prefix + '.txt', 'w') as f:
        for sentence in s2twp_corpus:
            f.write(sentence + '\n')


def random_replace_word(text, tokenizer, chinese_token_ids):
    try:
        segmented_text_list = list(text)
        text_token_ids = tokenizer.convert_tokens_to_ids(segmented_text_list)

        rand_idx = np.random.randint(len(text_token_ids))
        text_token_ids[rand_idx] = random.choice(chinese_token_ids)

        replaced_text = tokenizer.convert_ids_to_tokens(text_token_ids)
        replaced_text = tokenizer.convert_tokens_to_string(
            replaced_text).replace(' ', '')

        return replaced_text
    except:
        print(text)


def random_delete_word(text):
    try:
        rand_idx = np.random.randint(len(text))
        deleted_text = text[:rand_idx] + text[rand_idx+1:]

        return deleted_text
    except:
        print(text)


def make_noises(text, tokenizer, chinese_token_ids):
    noises_count = np.random.randint(10)
    noise_text_list = []

    for _ in range(noises_count):
        noise_text = text
        is_make_noise = bool(np.random.choice(np.arange(2), p=[0.9, 0.1]))
        one_more = True
        while one_more:
            noise_mode = np.random.randint(2)
            if noise_mode == 0:
                noise_text = random_replace_word(
                    noise_text, tokenizer, chinese_token_ids)
            elif len(noise_text) > 2:
                noise_text = random_delete_word(noise_text)
            else:
                break
            one_more = bool(np.random.choice(np.arange(2), p=[0.8, 0.2]))

        noise_text_list.append(noise_text)

    return noise_text_list


tokenizer = BertTokenizerFast.from_pretrained('ckiplab/bert-base-chinese')
chinese_token_ids = list(range(670, 7992))

corpus_topic = ['stock', 'finance']
for topic in corpus_topic:
    with open('./data_preprocess/' + topic + '.txt', 'r') as f:
        corpus = f.read().replace(' ', '').splitlines()

    noise_corpus_dict = {}
    for idx, sentence in tqdm(enumerate(corpus), total=len(corpus)):
        noise_sentence_data = {}
        noise_sentence_data['noise_id'] = idx
        noise_sentence_data['ground_truth_sentence'] = sentence
        noise_sentence_data['noise_sentence_list'] = make_noises(
            sentence, tokenizer, chinese_token_ids)

        noise_corpus_dict[idx] = noise_sentence_data

    with open('./data_preprocess/noise_' + topic + '.json', 'w') as f:
        json.dump(noise_corpus_dict, f)

    with open('./data_preprocess/noise_' + topic + '.json', 'r') as f:
        noise_data = json.load(f)

    id_list = []
    ground_truth_sentence_list = []
    converted_sentence_list = []
    for idx, noise_dict in tqdm(noise_data.items()):
        noise_sentence_len = len(noise_dict['noise_sentence_list'])
        id_list.extend([noise_dict['noise_id']] * noise_sentence_len)
        ground_truth_sentence_list.extend(
            [noise_dict['ground_truth_sentence']] * noise_sentence_len)
        converted_sentence_list.extend(noise_dict['noise_sentence_list'])

    noise_corpus_data = {'id': id_list,
                         'ground_truth_sentence': ground_truth_sentence_list,
                         'converted_sentence': converted_sentence_list}

    output_df = pd.DataFrame(data=noise_corpus_data)
    output_df.to_csv('./data_preprocess/noise_' + topic + '.csv', index=False)
