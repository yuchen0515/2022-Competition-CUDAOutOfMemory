import math


def unigram_perplexity(sentence_list, ws, uni_gram_dict, k=0.0001):
    segmented_sentence_list = ws(sentence_list)
    vocab_sum = sum(uni_gram_dict.values())
    vocab_len = len(uni_gram_dict)
    k = k
    perplexity_list = []
    for segmented_sentence in segmented_sentence_list:
        sentence_len = len(segmented_sentence)
        p = 1
        for i in range(len(segmented_sentence)):
            p *= ((uni_gram_dict.get(segmented_sentence[i], 0) + k) / (
                vocab_sum + k*vocab_len))

        p = pow(p, -1/sentence_len)
        perplexity_list.append(p)

    return perplexity_list


def bigram_perplexity(sentence_list, ws, uni_gram_dict, bi_gram_dict, k=0.0001):
    segmented_sentence_list = ws(sentence_list)

    vocab_sum = sum(uni_gram_dict.values())
    vocab_len = len(uni_gram_dict)
    k = k  # smoothing

    perplexity_list = []
    for segmented_sentence in segmented_sentence_list:
        sentence_len = len(segmented_sentence)
        p = 1
        if sentence_len > 0:
            p = (uni_gram_dict.get(
                segmented_sentence[0], 0) + k) / (vocab_sum + k*vocab_len)
            for i in range(1, sentence_len):
                p = p * ((bi_gram_dict.get(' '.join(segmented_sentence[i-1:i+1]), 0) + k) / (
                    uni_gram_dict.get(segmented_sentence[i-1], 0) + k * vocab_len))
            if p != 0:
                p = math.log(pow(p, -1/sentence_len))
            else:
                p = math.log(100000)
        else:
            p = math.log(100000)

        perplexity_list.append(p)

    return perplexity_list


def trigram_perplexity(sentence_list, ws, uni_gram_dict, bi_gram_dict, tri_gram_dict, k=0.0001):
    segmented_sentence_list = ws(sentence_list)

    vocab_sum = sum(uni_gram_dict.values())
    vocab_len = len(uni_gram_dict)
    k = k  # smoothing

    perplexity_list = []
    for segmented_sentence in segmented_sentence_list:
        sentence_len = len(segmented_sentence)

        p = 1
        if sentence_len == 0:
            p = 0
        if sentence_len > 0:
            p = p * \
                (uni_gram_dict.get(
                    segmented_sentence[0], 0) + k) / (vocab_sum + k*vocab_len)
        if sentence_len > 1:
            p = p * ((bi_gram_dict.get(' '.join(segmented_sentence[0:2]), 0) + k) / (
                uni_gram_dict.get(segmented_sentence[0], 0) + k * vocab_len))
            for i in range(2, sentence_len):
                p = p * (tri_gram_dict.get(' '.join(segmented_sentence[i-2:i+1]), 0) + k) / (
                    bi_gram_dict.get(' '.join(segmented_sentence[i-1:i+1]), 0) + k*vocab_len)

        if p != 0:
            p = math.log(pow(p, -1/sentence_len))
        else:
            p = math.log(100000)

        perplexity_list.append(p)

    return perplexity_list
