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

##################### Embedding layer ###################


class Embedding(nn.Module):
    '''
    Embedding of words and positional Encoding 
    '''
    def __init__(self, vocab_size, d_model, max_len = config.max_len):

        super(Embedding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(0.1)
        ## embedding layer of size vocab_size * d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = self.create_pe(max_len, self.d_model)
    
    def create_pe(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model).to(device)
        for pos in range(max_len):   # for each position of the word
            for i in range(0, d_model, 2):   # for each dimension of the each position
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)   # include the batch size
        return pe

    def forward(self, encoded_words):
        ## Giving higher weight to the embedding
        embedding = self.embed(encoded_words) * math.sqrt(self.d_model)
        embedding += self.pe[:,:embedding.size(1)]
        embedding = self.dropout(embedding)
        return embedding





######################Attention Layer########################
class Attention(nn.Module):

    def __init__(self, heads, d_model):

        super(Attention,self).__init__()
        ### Check whether heads divide d_model evenly
        assert d_model % heads == 0
        self.d_h = d_model // heads
        self.d_model = d_model
        self.heads = heads
        self.dropout = nn.Dropout(0.1)
        self.query = nn.Linear(d_model,d_model)
        self.key = nn.Linear(d_model,d_model)
        self.value = nn.Linear(d_model,d_model)
        self.concat = nn.Linear(d_model,d_model)

    def forward(self,query,key,value,mask):
        '''
        query,key and value of shape batch_size * max_len * dimensionality
        mask of shape: (batch_size,1,1, max_words)
        '''
        ## Weights of same dimension so make sure dimensionality 
        ## doesn't change
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # permute function used to alter dimension
        # batch_size ,max_len,word_dimension -> batch_size,max_len,h,d_h -> batch_size,h,max_len,d_h
        batch_size, max_len,_ = query.shape

        ## let's break these into chunks for heads
        query = query.view(batch_size,-1,self.heads,self.d_h).permute(0,2,1,3)
        key = key.view(batch_size,-1,self.heads,self.d_h).permute(0,2,1,3)
        value = value.view(batch_size,-1,self.heads,self.d_h).permute(0,2,1,3)

        ## let's calculate attention yo!
        ## We will get batch_size,h ,max_len,max_len
        ## Getting scores and normalizing
        scores = torch.matmul(query,key.permute(0,1,3,2)) / math.sqrt(self.d_h)

        ## For the decoder where the mask is 0
        scores = scores.masked_fill(mask == 0,-1e9)
        ## Take the softmax to obtain the weights
        weights = F.softmax(scores,dim = -1)
        weights = self.dropout(weights)

        ## To obtain the context multiply with value
        context = torch.matmul(weights,value)
        ## remember contiguous makes the copy of the array
        context = context.permute(0,2,1,3).contiguous().view(batch_size,max_len,self.d_model)

        ## Pass it through concat layer which is
        ## again of same dimensionality as the embedding dimension
        interacted = self.concat(context)
        return interacted


####################### Feed Forward Layer ################

class FeedForward(nn.Module):
    '''
    Takes the input as dimensionality of word embedding
    and hidden dimension size
    '''
    def __init__(self, d_model, h_dim = config.hidden_dim):

        super(FeedForward,self).__init__()

        self.fc1 = nn.Linear(d_model,h_dim)
        self.fc2 = nn.Linear(h_dim,d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self,x):

        out = F.relu(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out



####################### Encoder Layer ##########################

class Encoder(nn.Module):

    def __init__(self,d_model,heads):

        super(Encoder,self).__init__()

        self.layernorm = nn.LayerNorm(d_model)
        self.s_attention = Attention(heads,d_model)
        self.feed_forward = FeedForward(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, embedding,mask):
        
        ## Self attention of encoder
        interacted = self.s_attention(embedding,embedding,embedding,mask)
        interacted = self.dropout(interacted)

        ## Applying layernormalization and residual learning
        interacted = self.layernorm(interacted + embedding)
        feed_forward_out = self.feed_forward(interacted)
        feed_forward_out = self.dropout(feed_forward_out)
        encoded = self.layernorm(feed_forward_out + interacted)
        return encoded



################################### Decoder Layer #######################
class Decoder(nn.Module):

    def __init__(self,d_model,heads):
        super(Decoder,self).__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.s_attention = Attention(heads,d_model)
        self.m_attention = Attention(heads,d_model)
        self.feed_forward = FeedForward(d_model)
        self.dropout = nn.Dropout(0.1)

    
    def forward(self,embedding,encoded,src_mask,target_mask):
        
        ### Self attention of decoder
        context = self.s_attention(embedding,embedding,embedding,target_mask)
        context = self.dropout(context)
        context = self.layernorm(context + embedding)

        ## Encoder Decoder Attention
        interacted = self.m_attention(context, encoded, encoded, src_mask)
        interacted = self.dropout(interacted)


        feed_out = self.feed_forward(interacted)
        feed_out = self.dropout(feed_out)

        decoded = self.layernorm(feed_out + interacted)

        return decoded




######################### Finaaaly Transformer Layer ! #############
    
class Transformer(nn.Module):

    def __init__(self, d_model, heads, num_layer, word_map):

        super(Transformer, self).__init__()

        self.d_model = d_model
        self.vocab_size = len(word_map)
        self.embed = Embedding(self.vocab_size,d_model)
        ### There are num_layers of encoders and decoders
        self.encoder = nn.ModuleList([Encoder(d_model, heads) for _ in range(num_layer)])
        self.decoder = nn.ModuleList([Decoder(d_model, heads) for _ in range(num_layer)])
        self.logit = nn.Linear(d_model, self.vocab_size)

    def encode(self, src_words, src_mask):
        ### Obtain embedding and pass it through encoder layer
        src_embeddings = self.embed(src_words)
        for layer in self.encoder:
            src_embeddings = layer(src_embeddings, src_mask)
        return src_embeddings
    
    def decode(self, target_words, target_mask, src_embeddings, src_mask):
        ### Obtain embedding and pass it through decoder layer
        tgt_embeddings = self.embed(target_words)
        for layer in self.decoder:
            tgt_embeddings = layer(tgt_embeddings, src_embeddings, src_mask, target_mask)
        return tgt_embeddings
        
    def forward(self, src_words, src_mask, target_words, target_mask):
        ## Encoder Decoder and then softmax
        encoded = self.encode(src_words, src_mask)
        decoded = self.decode(target_words, target_mask, encoded, src_mask)
        out = F.log_softmax(self.logit(decoded), dim = 2)
        return out        


    