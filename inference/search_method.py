import re
import torch
from torch.utils.data import Dataset, DataLoader
from einops import rearrange


def generate_mask(length):
    mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
    mask = mask.float().masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, float(0.0))

    return mask


def greedy_search(model,
                  tokenizer,
                  device,
                  batch,
                  batch_size,
                  max_len: int = 128,
                  cls_token_id: int = 101,
                  sep_token_id: int = 102,
                  pad_token_id: int = 0):
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
