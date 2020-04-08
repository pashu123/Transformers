import torch
import torch.nn as nn
import math
import torch.nn.functional as F

## Select the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################## Embedding layer ###################
## 1.Embedding for words
## 2. Positional Encoding

class Embedding(nn.Module):
    '''
    Embedding of words and positional Encoding 
    '''
    def __init__(self, vocab_size, d_model, max_len):

        super(Embedding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(0.1)
        ## embedding layer of size vocab_size * d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = self.create_pe(max_len, self.d_model)
    
    def create_pe(self, max_len, d_model):
        '''
        Creates the positional encoding of the sequence
        '''
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * \
         (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

    def forward(self, encoded_words):
        ## Giving higher weight to the embedding
        embedding = self.embed(encoded_words) * math.sqrt(self.d_model)
        embedding += self.pe[:,:embedding.size(1)]
        embedding = self.dropout(embedding)
        return embedding


###########################################################################


######################Attention Layer########################
class Attention(nn.Module):

    def __init__(self, heads, d_model):

        super(Attention,self).__init__()
        ### Check whether heads divide d_model evenly
        assert d_model % heads == 0
        self.d_h = d_model // heads
        self.heads = heads
        self.dropout = nn.Dropout(0.1)
        self.query = nn.Linear(d_model,d_model)
        self.key = nn.Linear(d_model,d_model)
        self.value = nn.Linear(d_model,d_model)
        self.concat = nn.Linear(d_model,d_model)

        

    
        



