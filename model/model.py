import math
import torch
import torch.nn as nn
from torch.nn.modules import transformer


class PositionEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 奇數
        pe[:, 0::2] = torch.sin(position * div_term)
        # 偶數
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Embedding(nn.Module):
    def __init__(self, d_model, num_of_vocab):
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(num_of_vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class GecTransformer(nn.Module):
    def __init__(self, max_len: int = 200, num_of_vocab: int = 22399, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 4,
                 num_decoder_layers: int = 4, dim_feedforward: int = 1024, dropout: float = 0.1,
                 activation: str = "relu"):
        super(GecTransformer, self).__init__()

        # encoder
        encoder_norm = transformer.LayerNorm(d_model)
        encoder_layer = transformer.TransformerEncoderLayer(d_model,
                                                            nhead,
                                                            dim_feedforward,
                                                            dropout,
                                                            activation)
        self.encoder = transformer.TransformerEncoder(encoder_layer,
                                                      num_encoder_layers,
                                                      encoder_norm)

        # decoder
        decoder_norm = transformer.LayerNorm(d_model)
        decoder_layer = transformer.TransformerDecoderLayer(d_model,
                                                            nhead,
                                                            dim_feedforward,
                                                            dropout,
                                                            activation)
        self.decoder = transformer.TransformerDecoder(decoder_layer,
                                                      num_decoder_layers,
                                                      decoder_norm)
        #
        self.embedding = Embedding(d_model, num_of_vocab)
        self.position_encoding = PositionEncoding(
            d_model=d_model, max_len=max_len)
        self.projection = nn.Linear(d_model, num_of_vocab, bias=False)

    def forward(self, src, trg, src_key_padding_mask, trg_key_padding_mask, src_mask, trg_mask, memory_key_padding_mask):
        src_embedding = self.embedding(src)
        src_embedding = self.position_encoding(src_embedding)
        src_embedding = src_embedding.permute(1, 0, 2)

        memory = self.encoder(src_embedding,
                              mask=src_mask,
                              src_key_padding_mask=src_key_padding_mask)

        trg_embedding = self.embedding(trg)
        trg_embedding = self.position_encoding(trg_embedding)
        trg_embedding = trg_embedding.permute(1, 0, 2)

        decoder_output = self.decoder(trg_embedding,
                                      memory,
                                      tgt_mask=trg_mask,
                                      tgt_key_padding_mask=trg_key_padding_mask,
                                      memory_key_padding_mask=memory_key_padding_mask)
        decoder_output = decoder_output.permute(1, 0, 2)
        output = self.projection(decoder_output)
        return output
