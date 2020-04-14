import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import config
import copy
import math

## Select the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############# Helper Functions ################################


def get_clones(module, N):
    '''Creates clones of N encoder and decoder layers'''
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def scaled_dot_product_attention(query, key, value, mask):
    ''' query, key, value : batch_size * heads * max_len * d_h
        return output : batch_size * heads * max_len * d_h
    '''
    
    matmul = torch.matmul(query,key.transpose(-2,-1))
    scale = torch.tensor(query.shape[-1],dtype=float)
    logits = matmul / torch.sqrt(scale)
    if mask is not None:
        logits += (mask.float() * -1e9)
    
    attention_weights = F.softmax(logits,dim = -1)
    output = torch.matmul(attention_weights,value)
    return output


def create_padding_mask(x):
    '''creates padding mask so that 
        padding doesn't contribute to overall loss
        return mask of shape (batch_size,1,1,max_len)'''
    mask = (x == 0) * 1
    mask = mask.unsqueeze(1).unsqueeze(1)
    return mask

def create_look_ahead_mask(x):
    '''to create look_ahead mask for output so as to see 
        previous word to predict the next one,also creates
        mask for padding data.
        mask of shape (batch_size * 1 * max_len * max_len)'''

    seq_len = x.shape[1]
    mask = torch.triu(torch.ones(seq_len, seq_len)).transpose(0, 1).type(dtype=torch.uint8)
    mask = mask.to(device)
    mask = (mask == 0) * 1
    mask = mask.unsqueeze(0)
    pad_mask = create_padding_mask(x)
    return torch.max(mask,pad_mask)



############# We will break the model into 6 Subparts #############
## 1. Embedding Class  (Embedder)
## 2. Attention Class (Multihead Attention and helper scaled_dot_product_attention)
## 3. Feed Forward Class (Feed Forward neural net)
## 4. Encoder Class (Encoder layer and Encoder)
## 5. Decoder Class (Decoder layer and Decoder)
## 6. Transformer Class (Finally Transformer)


class Embedder(nn.Module):
    '''Input embedding layer of size vocab_size * dimensionality
    of word embedding'''
    def __init__(self,vocab_size,d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    
    def forward(self,x):
        return self.embed(x)


class PositionalEncoding(nn.Module):
    '''Transformers are not sequential so positional encoding
    gives some sequentiality to sentence'''

    def __init__(self, d_model, dropout=0.1, max_len=40):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() \
                            * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe

    def forward(self, x):
        x *= math.sqrt(self.d_model)
        x +=  self.pe[:,:x.size(1)]
        return self.dropout(x)


class FeedForward(nn.Module):
    '''Feed Forward neural network, simple isn't it'''
    def __init__(self,d_model,d_ff = 2048,dropout = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self,x):
        x = F.relu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x







class MultiHeadAttention(nn.Module):
    '''Divides d_model into heads and
    applies attention to each layer with helper 
    function scaled_dot_product_attention'''

    def __init__(self, heads, d_model):
        super().__init__()
        self.heads = heads
        self.d_model = d_model

        assert d_model % self.heads == 0

        self.d_h = self.d_model // self.heads

        self.q_dense = nn.Linear(d_model,d_model)
        self.k_dense = nn.Linear(d_model,d_model)
        self.v_dense = nn.Linear(d_model,d_model)

        self.out = nn.Linear(d_model,d_model)

    
    def forward(self, q, k, v, mask = None):
        
        # batch_size
        bs = q.size(0)

        k = self.k_dense(k).view(bs, -1, self.heads, self.d_h)
        q = self.q_dense(q).view(bs, -1, self.heads, self.d_h)
        v = self.v_dense(v).view(bs, -1, self.heads, self.d_h)

        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        scores = scaled_dot_product_attention(q,k,v,mask)
        
        # concat each heads
        concat = scores.transpose(1,2).contiguous()\
            .view(bs,-1,self.d_model)
        
        out = self.out(concat)

        return out




class EncoderLayer(nn.Module):
    '''Encoder layer of transformer 
    embedding -> positional_encoding -> attention
     -> Feed Forward with some resnet connection'''
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    
    def forward(self,x,mask):
        x1 = self.norm_1(x)
        x1 = x + self.dropout_1(self.attn(x1,
                                x1,x1,mask))
        x2 = self.norm_2(x1)
        x3 = x1 + self.dropout_2(self.ff(x2))

        return x3


class DecoderLayer(nn.Module):
    '''Takes one input from encoder and another from out target words'''
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads,d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x, encoder_out, src_mask, trg_mask):

        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, encoder_out, encoder_out,
        src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


class Encoder(nn.Module):
    '''Cloning and making copies'''
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    

class Decoder(nn.Module):
    '''Cloning and making copies'''
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

    
class Transformer(nn.Module):
    '''Finally Transformer yup done!'''
    def __init__(self, vocab_size, d_model, num_layers, heads):
        super().__init__()
        self.encoder = Encoder(vocab_size,d_model,num_layers,heads)
        self.decoder = Decoder(vocab_size,d_model,num_layers,heads)
        self.out = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, trg, src_mask, trg_mask):

        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output


