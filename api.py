from argparse import ArgumentParser
from flask import Flask
from flask import request
from flask import jsonify

from transformers import BertModel, BertTokenizerFast
from ckiptagger import WS

import datetime
import hashlib
import random
import json
import time
import random

import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import numpy as np

from inference.config import *
from inference.inference import *
from inference.model import *

# Other tools
from nlp_fluency.models import NgramsLanguageModel
from service_streamer import ThreadedStreamer

# Pre-load model
app = Flask(__name__)

model_path = r"./trained_model/Stock-Origin_epoch4.bin"
tokenizer_path = r"bert-base-chinese"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
ws_path = r"./data"
ngram_dict_dir = r"./ngram_lm"
trigram_lm_dir = r"./ngram_lm/trigram"

tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
model = torch.load(model_path, map_location=device)

with open("{}/unigram.json".format(ngram_dict_dir)) as dict:
    tri_gram_dict = json.load(dict)

trigram_lm = NgramsLanguageModel.from_pretrained(trigram_lm_dir)
ws = WS(ws_path)

inference = Inference(model, tokenizer, device, Config)

streamer = ThreadedStreamer(
    inference.only_inference, batch_size=64, max_latency=0.15)


####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = 'i98565412@gmail.com'          #
SALT = 'mirlabOwenNAVI'                        #
#########################################


def generate_server_uuid(input_string):
    """ Create your own server_uuid.

    @param:
        input_string (str): information to be encoded as server_uuid
    @returns:
        server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string + SALT).encode("utf-8")
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid


def predict(sentence_list, phoneme_sequence_list, retry):
    """ Predict your model result.

    @param:
        sentence_list (list): an list of sentence sorted by probability.
        phoneme_sequence_list (list): an list of phoneme sequence sorted by probability.
    @returns:
        prediction (str): a sentence.
    """

    ####### PUT YOUR MODEL INFERENCING CODE HERE #######
    prediction = ""
    TOP1 = sentence_list[0].replace(' ', '')
    corrected_sentence_list = [TOP1]

    print(len(sentence_list[0]))

    if retry == 2 and len(sentence_list[0]) <= 40:
        sentence_list = sentence_list[:1]
        corrected_list_temp = streamer.predict(sentence_list)
        corrected_sentence_list.append(corrected_list_temp[0])
        print("test: correct", corrected_sentence_list)

        corrected_perpelxity_list = []
        for corrected_sentence in corrected_sentence_list:
            corrected_perpelxity_list.append(
                trigram_lm.perplexity(ws([corrected_sentence])[0]))

        ranking = np.argsort(corrected_perpelxity_list)
        prediction = corrected_sentence_list[ranking[0]]
        print("transformer, {}".format(prediction))

    else:
        prediction = TOP1
        if retry < 2:
            print("retry < 2, {}".format(prediction))
        else:
            print("retry = 2, len >= 25, {}".format(prediction))

    ####################################################

    try:
        if _check_datatype_to_string(prediction):
            return prediction
    except:
        return TOP1

    return TOP1


def _check_datatype_to_string(prediction):
    """ Check if your prediction is in str type or not.
        If not, then raise error.

    @param:
        prediction: your prediction
    @returns:
        True or raise TypeError.
    """
    if isinstance(prediction, str):
        return True
    raise TypeError('Prediction is not in string type.')


@app.route('/inference', methods=['POST'])
def inference():
    """ API that return your model predictions when E.SUN calls this API. """
    data = request.get_json(force=True)

    # 自行取用，可紀錄玉山呼叫的 timestamp
    esun_timestamp = data['esun_timestamp']

    # 取 sentence list 中文
    sentence_list = data['sentence_list']
    # 取 phoneme sequence list (X-SAMPA)
    phoneme_sequence_list = data['phoneme_sequence_list']

    retry = int(data['retry'])

    t = datetime.datetime.now()
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL + ts)

    try:
        answer = predict(sentence_list, phoneme_sequence_list, retry)
    except TypeError as type_error:
        # You can write some log...
        raise type_error
    except Exception as e:
        # You can write some log...
        raise e
    server_timestamp = time.time()

    return jsonify({'esun_uuid': data['esun_uuid'],
                    'server_uuid': server_uuid,
                    'answer': answer,
                    'server_timestamp': server_timestamp})


@app.route('/')
def start():
    return "Hello World"


if __name__ == "__main__":
    arg_parser = ArgumentParser(
        usage='Usage: python ' + __file__ + ' [--port <port>] [--help]'
    )
    arg_parser.add_argument('-p', '--port', default=614, help='port')
    arg_parser.add_argument('-d', '--debug', default=False, help='debug')
    options = arg_parser.parse_args()

    app.run(host='0.0.0.0', port=options.port,
            debug=options.debug, threaded=True, processes=True)
