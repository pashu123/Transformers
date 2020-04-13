import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import config

############# We will break the model into 6 Subparts #############
## 1. Embedding Class 
## 2. Attention Class
## 3. Feed Forward Class
## 4. Encoder Class
## 5. Decoder Class
## 6. Transformer Class


## Select the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

      



def scaled_dot_product_attention(query, key, value, mask):
    '''query, key, value : batch_size * heads * max_len * d_h
        return output : batch_size * heads * max_len * d_h
    '''
    
    matmul = torch.matmul(query,key.transpose(-2,-1))

    scale = torch.tensor(query.shape[-1],dtype=float)

    logits = matmul / torch.sqrt(scale)

    if mask is not None:
        logits += (mask * -1e9)
    
    attention_weights = F.softmax(logits,dim = -1)

    output = torch.matmul(attention_weights,value)

    return output


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.h = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.d_h = d_model // self.num_heads

        self.q_dense = nn.Linear(d_model,d_model)
        self.k_dense = nn.Linear(d_model,d_model)
        self.v_dense = nn.Linear(d_model,d_model)

        self.out = nn.Linear(d_model,d_model)

    
    def forward(self, q, k, v, mask = None):
        
        # batch_size
        bs = q.size(0)

        k = self.k_dense(k).view(bs, -1, self.h, self.d_h)
        q = self.q_dense(q).view(bs, -1, self.h, self.d_h)
        v = self.v_dense(v).view(bs, -1, self.h, self.d_h)

        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        scores = scaled_dot_product_attention(q,k,v,mask)
        
        # concat each heads
        concat = scores.transpose(1,2).contiguous()\
            .view(bs,-1,self.d_model)
        
        out = self.out(concat)

        return out



def create_padding_mask(x):

    mask = (x == 0) * 1
    mask = mask.unsqueeze(1).unsqueeze(1)
    return mask


def create_look_ahead_mask(x):

    seq_len = x.shape[1]
    mask = torch.triu(torch.ones(seq_len, seq_len)).transpose(0, 1).type(dtype=torch.uint8)
    mask = mask.to(device)
    mask = (mask == 0) * 1
    mask = mask.unsqueeze(0)
    pad_mask = create_padding_mask(x)
    return torch.max(mask,pad_mask)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe

    def forward(self, x):
        x +=  self.pe
        return self.dropout(x)


class FeedForward(nn.Module):
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




    
    





query = np.array([[[3,4],[3,4],[4,3]]], dtype = float)
key = np.array([[[3,4],[3,4],[4,3]]], dtype = float)
value = np.array([[[3,4],[3,4],[4,3]]], dtype = float)
query = torch.from_numpy(query)
key = torch.from_numpy(key)
value = torch.from_numpy(value)
print(query.shape)
mask = None

scaled_dot_product_attention(query,key,value,mask)

    