import numpy as np
import re

from evaluate import load
from tqdm import tqdm
from ckiptagger import WS

from torch.utils.data import Dataset, DataLoader
from einops import rearrange

from inference.config import Config
from inference.perplexity import *

from nlp_fluency.models import NgramsLanguageModel
from model.data import *
from model.model import *


def generate_mask(length):
    mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
    mask = mask.float().masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, float(0.0))

    return mask


def greedy_search(model, tokenizer, device,
                  batch, batch_size,
                  max_len: int = 128,
                  cls_token_id: int = 101, sep_token_id: int = 102, pad_token_id: int = 0):
    model.eval()

    src_input_ids = batch[0].to(device)
    src_key_padding_mask = batch[1].to(device)
    src_mask = generate_mask(src_input_ids.shape[1]).to(device)
    memory_key_padding_mask = src_key_padding_mask

    corrected_sentence_list = []
    is_done = [False] * batch_size
    predict_sentence_ids = torch.ones(batch_size, 1).fill_(
        cls_token_id).type_as(src_input_ids.data)
    for i in range(max_len - 1):
        trg_mask = generate_mask(predict_sentence_ids.shape[1]).to(device)
        output = model.forward(src_input_ids,
                               predict_sentence_ids,
                               src_key_padding_mask=src_key_padding_mask,
                               trg_key_padding_mask=None,
                               src_mask=src_mask,
                               trg_mask=trg_mask,
                               memory_key_padding_mask=memory_key_padding_mask)
        output = output[:, -1, :]
        _, next_word = torch.max(output, dim=1)
        #next_word = next_word.data[0]
        # 101=[CLS] 102 = [SEP]
        predict_sentence_ids = torch.cat(
            [predict_sentence_ids, next_word.view(-1, 1)], dim=1)
        for i in range(len(next_word)):
            if next_word[i] == sep_token_id:
                is_done[i] = True

        if all(is_done):
            break

    for ids in predict_sentence_ids:
        tokens = ''.join(tokenizer.convert_ids_to_tokens(ids))
        tokens = re.sub(r'\[CLS]|\[SEP]|\[PAD]\[UNK]', '', tokens)
        corrected_sentence_list.append(tokens)

    return corrected_sentence_list

# Scoring


def penalty_func(src_len, trg_len, first_penalty=0.25, second_penalty=1):
    diff = abs(src_len - trg_len)
    if diff == 1:
        return first_penalty
    elif diff == 2:
        return second_penalty
    return diff * diff


def get_scores(sentence_list, phoneme_length_list,
               ws, uni_gram_dict, bi_gram_dict, tri_gram_dict,
               nlp_fluency_lm,
               length_penalty=True,
               first_penalty=0.25, second_penalty=1):

    if nlp_fluency_lm is not None:
        segmented_sentence_list = ws(sentence_list)
        perplexity_list = []
        for sentence in segmented_sentence_list:
            perplexity = math.log(nlp_fluency_lm.perplexity(sentence))
            perplexity_list.append(perplexity)
    else:
        perplexity_list = trigram_perplexity(
            sentence_list, ws, uni_gram_dict, bi_gram_dict, tri_gram_dict)

    score_list = []
    for sentence, perplexity, phoneme_length in zip(sentence_list, perplexity_list, phoneme_length_list):
        if length_penalty:
            score = perplexity + penalty_func(len(sentence), phoneme_length)
        else:
            score = perplexity
        score_list.append(score)

    return score_list


# Evaluation
def evaluation(model_or_model_path, tokenizer, device,
               dev_data,
               ws, uni_gram_dict, bi_gram_dict, tri_gram_dict,
               nlp_fluency_lm,
               length_penalty=True,
               first_penalty=0.25,
               second_penalty=1,
               anchor_cer_threshold=-1):
    if isinstance(model_or_model_path, str):
        model = torch.load(model_or_model_path, map_location=device)
    elif isinstance(model_or_model_path, GecTransformer):
        model = model_or_model_path

    model.to(device)
    model.eval()

    cer = load("cer")

    cer_list = []

    dev_id_list = dev_data['id'].unique()

    id_list = []
    ground_truth_list = []
    raw_sentence_list = []
    generated_sentence_list = []
    best_sentence_list = []
    with torch.no_grad():
        with tqdm(total=len(dev_id_list)) as pbar:
            pbar.set_description('Evaluation')

            for idx, dev_id in enumerate(dev_id_list):
                sentence_list = list(
                    dev_data[dev_data['id'] == dev_id]['converted_sentence'].values)
                ground_truth = dev_data[dev_data['id'] == dev_id]['ground_truth_sentence'].unique()[
                    0]
                phoneme_length_list = dev_data[dev_data['id']
                                               == dev_id]['phoneme_length'].values

                sentence_list = [sentence_list[0]]
                dev_dataset = GecDatasetInference(
                    sentence_list, tokenizer, Config)
                dev_dataloader = DataLoader(
                    dataset=dev_dataset, batch_size=dev_dataset.len, collate_fn=collate_fn_inference)
                corrected_sentence_list = []

                for batch in dev_dataloader:
                    corrected_sentence_list = greedy_search(model, tokenizer, device,
                                                            batch, dev_dataset.len)

                corrected_scores = get_scores(corrected_sentence_list, phoneme_length_list,
                                              ws, uni_gram_dict, bi_gram_dict, tri_gram_dict,
                                              nlp_fluency_lm)
                raw_scores = get_scores(sentence_list, phoneme_length_list,
                                        ws, uni_gram_dict, bi_gram_dict, tri_gram_dict,
                                        nlp_fluency_lm,
                                        length_penalty)

                all_sentence_list = sentence_list + corrected_sentence_list
                all_scores_list = np.array((raw_scores + corrected_scores))
                best_sentence = all_sentence_list[np.argmin(all_scores_list)]

                #=====================================#
                # id_list.append(dev_id)
                ground_truth_list.append(ground_truth)
                # raw_sentence_list.append(sentence_list[0])
                # generated_sentence_list.append(corrected_sentence_list[0])

                best_sentence_list.append(best_sentence)

                pbar.update(1)
                if (idx+1) % 100 == 0:
                    curr_cer = cer.compute(
                        predictions=best_sentence_list, references=ground_truth_list)
                    pbar.set_postfix(average_CER=curr_cer)

    average_CER = cer.compute(
        predictions=best_sentence_list, references=ground_truth_list)
    return average_CER
