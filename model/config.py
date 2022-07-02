import torch

#####################CONFIG#####################
class Config:
    model_path = 'bert-base-chinese'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    ngram_dict_dir = '../ngram_lm'
    
    vocab_size = 21128
    max_length = 200
    
    batch_size = 28
    lr = 0.0001
    epoch = 5
    max_norm = 1.0
    
    model_dir = 'test_model_1'
    
################################################