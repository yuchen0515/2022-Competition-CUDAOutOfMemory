import torch
import pandas as pd
import random
from torch.utils.data import Dataset


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


def collate_fn_inference(batch):
    src_input_ids, src_key_padding_mask, src_sequence_len = map(
        torch.stack, zip(*batch))

    src_max_len = max(src_sequence_len).item()
    src_input_ids = src_input_ids[:, :src_max_len]
    src_key_padding_mask = src_key_padding_mask[:, :src_max_len]

    return src_input_ids, src_key_padding_mask, src_sequence_len
