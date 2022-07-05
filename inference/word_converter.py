import re


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


def word_converter(sentences):
    '''
        Convert all input sentecnes to correct input form

        param sentences: a list of strings

        return converted_sentences: a list of converted strings
    '''
    converted_sentences = []
    for sentence in sentences:
        sentence = num_convert(sentence)
        sentence = english_convert(sentence)
        sentence = special_word_convert(sentence)

        converted_sentences.append(sentence)

    return converted_sentences
