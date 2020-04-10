import torch
from torch.utils.data import Dataset
import json
import config




class Dataset(Dataset):

    def __init__(self):

        self.pairs = json.load(open('pairs_encoded.json'))
        self.dataset_size = len(self.pairs)

    def __getitem__(self, i):
        
        question = torch.LongTensor(self.pairs[i][0])
        reply = torch.LongTensor(self.pairs[i][1])
            
        return question, reply

    def __len__(self):
        return self.dataset_size


train_loader = torch.utils.data.DataLoader(Dataset(),
                                config.batch_size,
                                shuffle= True,
                                pin_memory=True)