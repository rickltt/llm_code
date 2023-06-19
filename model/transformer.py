import torch
import math
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def _clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, n_head = 8, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % n_head == 0                             
        self.d = d_model // n_head
        self.n_head = n_head
        self.linears = _clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.layer_norm = LayerNorm(d_model, eps=1e-6)  
        self.dropout = nn.Dropout(p=dropout)
    
    def attention(self, query, key, value, mask=None, dropout=None):

        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
    
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        residual = query

        # 1) Do all the linear projections in batch from d_model => h x d
        query, key, value = [l(x).view(nbatches, -1, self.n_head, self.d).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.n_head * self.d)
        x = self.linears[-1](x)
        x += residual
        x = self.layer_norm(x)

        return x

  
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_model, dim_feedforward = 2048, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, dim_feedforward) # position-wise
        self.w_2 = nn.Linear(dim_feedforward, d_model) # position-wise
        self.layer_norm = LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        # add & norm
        x += residual

        x = self.layer_norm(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, n_head = 8, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(d_model, n_head, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, dim_feedforward, dropout)

    def forward(self, x, mask = None):
        x = self.self_attn(x,x,x,mask)
        x = self.feed_forward(x)
        return x
    

class Encoder(nn.Module):
    def __init__(self, d_model, vocab_size, n_layers =6, dim_feedforward = 2048, n_head=8, dropout=0.1 ):
        super(Encoder, self).__init__()

        self.emb = Embeddings(d_model, vocab_size)
        self.pe = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, dim_feedforward, n_head, dropout)
            for _ in range(n_layers)])
        
        self.layer_norm = LayerNorm(d_model)

    def forward(self, x, mask=None):
        enc_output = self.dropout(self.pe(self.emb(x)))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, mask=mask)
        
        return enc_output
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, n_head, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # masked self-attention
        self.slf_attn = MultiHeadedAttention(d_model, n_head, dropout)
        # encoder-decoder attention
        self.enc_attn = MultiHeadedAttention(d_model, n_head, dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, dim_feedforward, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)

        # q用自己的, k和v是encoder的输出
        dec_output = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output
    

class Decoder(nn.Module):
    def __init__(self, d_model, vocab_size, n_layers = 6, dim_feedforward = 2048, n_head = 8, dropout=0.1):

        super().__init__()

        self.emb = Embeddings(d_model, vocab_size)
        self.pe = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(p=dropout)

        # 多个Decoder Layer叠加
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, dim_feedforward, n_head, dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, x_mask = None, dec_mask =None):

        # Embedding & Position encoding
        dec_output = self.dropout(self.pe(self.emb(x)))
        dec_output = self.layer_norm(dec_output)

        # Decoder Layers
        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_output, enc_output, slf_attn_mask=x_mask, dec_enc_attn_mask=dec_mask)

        return dec_output
    

class Transformer(nn.Module):
    def __init__(self, d_model, vocab_size, dim_feedforward=2048, n_layers=6, n_head=8, dropout=0.1):

        super().__init__()

        # Encoder
        self.encoder = Encoder(d_model, vocab_size, n_layers, dim_feedforward, n_head, dropout)
        # Decoder
        self.decoder = Decoder(d_model, vocab_size, n_layers, dim_feedforward, n_head, dropout)

        # xavier初始化
        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_seq = None, trg_seq = None):
        # encoder & decoder
        enc_output = self.encoder(src_seq)
        dec_output= self.decoder(trg_seq, enc_output)
        
        return dec_output
    
if __name__ == '__main__':
    bs = 4
    seq_len = 128
    d_model = 768
    vocab_size = 30522
    transformer = Transformer(d_model, vocab_size)

    src = torch.randint(0, vocab_size, (bs,seq_len)) # bs x seq_len
    tgt = torch.randint(0, vocab_size, (bs,seq_len))

    inputs = transformer(src, tgt)
    print(inputs.shape)