import json
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.utils.data
from model import Transformer
import re
from dataset import vocab_dict



rev_vocab = {v:k for k,v in vocab_dict.items()}

def preprocess_sentence(sentence):
  sentence = sentence.lower().strip()
  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]+', " ", sentence)
  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
  sentence = sentence.strip()
  # adding a start and an end token to the sentence
  return sentence


def evaluate(sentence,model,max_len = 40):

    model.eval()

    sentence = preprocess_sentence(sentence)
    sentence = [rev_vocab.get(word,0) for word in sentence]
    sentence = [rev_vocab['<START>']] + sentence + [rev_vocab['<END>']]
    sentence = torch.LongTensor(sentence).to(device)
    sentence = sentence.unsqueeze(0)

    sentence_mask = (sentence == 0) * 1
    sentence_mask.unsqueeze(1).unsqueeze(1)
    sentence_mask = sentence_mask.to(device)

    encoded = model.encoder(sentence,sentence_mask)

    start_word = torch.LongTensor([[rev_vocab['<START>']]]).to(device)

    for i in range(max_len-1):

        size = words.shape[1]
        target_mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
        target_mask = mask.to(device)
        target_mask = (mask == 0) * 1

        decoded = model.decode(words, encoded, sentence_mask, target_mask)
        
        out = model.out(decoded)
        out = F.softmax(out, dim = -1)

        val, ix = out[:,-1].data.topk(1)
        next_word = ix[0][0]

        if next_word == rev_vocab['<END>']:
            break

        words = torch.cat([start_word, torch.LongTensor([[next_word]]).to(device)],dim = 1)

    
    if words.dim() == 2:
        words = words.squeeze(0)
        words = words.tolist()

    sen_idx = [w for w in words if w not in {vocab_dict['<START>']}]
    sentence = ' '.join([rev_vocab[sen_idx[k]] for k in range(len(sen_idx))])
    
    return sentence


checkpoint = torch.load('checkpoint_0.pth.tar')
transformer = checkpoint['transformer']



while(1):
    question = input("Question: ")
    if question == 'quit':
        break
    reply = evaluate(question,transformer)
    print(reply)
    




