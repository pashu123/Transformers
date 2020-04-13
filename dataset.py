import torch
from torch.utils.data import Dataset
import config
import pickle


with open('preprocess_input/question.pkl','rb') as f:
    question = pickle.load(f)

with open('preprocess_input/answers.pkl','rb') as f:
    answers = pickle.load(f)

with open('preprocess_input/vocab.pkl','rb') as f:
    vocab_dict = pickle.load(f)

vocab_dict[32350] = '<START>'
vocab_dict[32351] = '<END>'



class Dataset(Dataset):

    def __init__(self):

        self.question = question
        self.answers = answers 
        self.dataset_size = len(self.question)

    def __getitem__(self, i):
        
        question = torch.LongTensor(self.question[i])
        reply = torch.LongTensor(self.answers[i])
        return question, reply

    def __len__(self):
        return self.dataset_size


train_loader = torch.utils.data.DataLoader(Dataset(),
                                config.batch_size,
                                shuffle= True,
                                pin_memory=True)