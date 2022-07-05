import torch
import pandas as pd
import random
from torch.utils.data import Dataset


class GecDataset(Dataset):
    def __init__(self, data, tokenizer, config):
        #super(GEC_Dataset, self).__init__()
        self.data = data
        self.config = config
        self.len = len(data)
        #self.src_sentences = list(data['converted_sentence'])
        #self.trg_sentences = list(data['ground_truth_sentence'])
        self.tokenizer = tokenizer

        self.processed_data = self.data_process(self.data)

    def data_process(self, data):
        print('='*10 + 'Data Process Start' + '='*10)
        processed_data = []
        for _, row in data.iterrows():
            src_sentence = row['converted_sentence']
            trg_sentence = row['ground_truth_sentence']

            src_input_ids, src_key_padding_mask, src_sequence_len = self.tokenize(
                src_sentence, padding=True)
            trg_input_ids, trg_key_padding_mask, trg_sequence_len = self.tokenize(
                trg_sentence, padding=True)

            src_sequence_len = torch.tensor(src_sequence_len, dtype=torch.long)
            trg_sequence_len = torch.tensor(trg_sequence_len, dtype=torch.long)

            #src: 0/1/2
            #trg: 3/4/5
            processed_data.append((src_input_ids, src_key_padding_mask, src_sequence_len,
                                  trg_input_ids, trg_key_padding_mask, trg_sequence_len))

        print('='*10 + 'Data Process Successfully' + '='*10)
        return processed_data

    def __len__(self):
        return len(self.data)

    def tokenize(self, sentence, padding=True):
        try:
            if padding:
                token = self.tokenizer(sentence,
                                       padding='max_length',
                                       max_length=self.config.max_length)
            else:
                token = self.tokenizer(sentence)

            input_ids = torch.tensor(token['input_ids'], dtype=torch.long)
            attention_mask = torch.tensor(
                token['attention_mask'], dtype=torch.long)
            key_padding_mask = attention_mask < 1
            # +2 => [CLS] [SEP]
            sequence_len = len(sentence) + 2
        except:
            print(sentence)

        return input_ids, key_padding_mask, sequence_len

    def __getitem__(self, item):
        #src: 0/1/2
        #trg: 3/4/5
        src_input_ids = self.processed_data[item][0]
        src_key_padding_mask = self.processed_data[item][1]
        src_sequence_len = self.processed_data[item][2]

        trg_input_ids = self.processed_data[item][3]
        trg_key_padding_mask = self.processed_data[item][4]
        trg_sequence_len = self.processed_data[item][5]

        return src_input_ids, src_key_padding_mask, src_sequence_len, trg_input_ids, trg_key_padding_mask, trg_sequence_len


class GecDatasetInference(Dataset):
    def __init__(self, sentence_list, tokenizer, Config):
        self.sentence_list = sentence_list
        self.tokenizer = tokenizer
        self.config = Config
        self.len = len(sentence_list)
        self.processed_data = self.sentence_process(sentence_list)

    def sentence_process(self, sentence_list):
        processed_data_list = []
        for sentence in sentence_list:
            input_ids, key_padding_mask, sequence_len = self.tokenize(sentence)
            sequence_len = torch.tensor(sequence_len, dtype=torch.long)

            processed_data_list.append(
                (input_ids, key_padding_mask, sequence_len))

        return processed_data_list

    def tokenize(self, sentence, padding=True):

        if padding:
            token = self.tokenizer(sentence,
                                   padding='max_length',
                                   max_length=self.config.max_length)
        else:
            token = self.tokenizer(sentence)

        input_ids = torch.tensor(token['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(
            token['attention_mask'], dtype=torch.long)
        key_padding_mask = attention_mask < 1
        # +2 => [CLS] [SEP]
        sequence_len = len(sentence) + 2

        return input_ids, key_padding_mask, sequence_len

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        input_ids = self.processed_data[item][0]
        key_padding_mask = self.processed_data[item][1]
        sequence_len = self.processed_data[item][2]

        return input_ids, key_padding_mask, sequence_len


def collate_fn(batch):
    src_input_ids, src_key_padding_mask, src_sequence_len, trg_input_ids, trg_key_padding_mask, trg_sequence_len = map(
        torch.stack, zip(*batch))
    src_max_len = max(src_sequence_len).item()
    # print(src_max_len)
    trg_max_len = max(trg_sequence_len).item()

    src_input_ids = src_input_ids[:, :src_max_len]
    trg_input_ids = trg_input_ids[:, :trg_max_len]

    src_key_padding_mask = src_key_padding_mask[:, :src_max_len]
    trg_key_padding_mask = trg_key_padding_mask[:, :trg_max_len]

    return src_input_ids, src_key_padding_mask, src_sequence_len, trg_input_ids, trg_key_padding_mask, trg_sequence_len


def collate_fn_inference(batch):
    src_input_ids, src_key_padding_mask, src_sequence_len = map(
        torch.stack, zip(*batch))

    src_max_len = max(src_sequence_len).item()
    src_input_ids = src_input_ids[:, :src_max_len]
    src_key_padding_mask = src_key_padding_mask[:, :src_max_len]

    return src_input_ids, src_key_padding_mask, src_sequence_len


def data_handler(data_path: str, output_path_list: list,
                 sample_or_split: str, sample_or_train_ratio: float = 0.8,
                 max_len: int = 200, shuffle=True,
                 extract_first_sentence_only=False):

    print(f'Loading Data from {data_path}...')
    data = pd.read_csv(data_path)
    data = data.dropna()
    data = data.drop(
        data[data['ground_truth_sentence'].str.len() > max_len].index)
    print('Done')

    if sample_or_split == 'sample':
        print('Sampling...')
        id_list = data['id'].unique()
        if shuffle:
            random.shuffle(id_list)

        sample_size = int(sample_or_train_ratio * len(id_list))

        if extract_first_sentence_only:
            output_data = data[data['id'].isin(id_list[:sample_size])].groupby(
                'id').first().reset_index()
        else:
            output_data = data[data['id'].isin(id_list[:sample_size])]

        output_data.to_csv(output_path_list[0], index=False)

        print('Finish')

        print('data Path:', data_path)
        print('data shape:', data.shape)
        print('sampled data shape', output_data.shape)
        print('='*20)
        return output_data

    elif sample_or_split == 'split':
        print('Spliting...')
        id_list = data['id'].unique()
        if shuffle:
            random.shuffle(id_list)

        train_size = int(sample_or_train_ratio * len(id_list))

        if extract_first_sentence_only:
            train_data = data[data['id'].isin(id_list[:train_size])].groupby(
                'id').first().reset_index()
            test_data = data[data['id'].isin(id_list[train_size:])].groupby(
                'id').first().reset_index()
        else:
            train_data = data[data['id'].isin(id_list[:train_size])]
            test_data = data[data['id'].isin(id_list[train_size:])]

        train_data.to_csv(output_path_list[0], index=False)
        test_data.to_csv(output_path_list[1], index=False)

        print('Finish')

        print('data Path:', data_path)
        print('data shape:', data.shape)
        print('train data shape', train_data.shape)
        print('test data shape', test_data.shape)
        print('='*20)
        return train_data, test_data
