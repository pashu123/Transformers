import json
import torch
from dataset import train_loader
from model import Transformer
from extra_optimization import *
import config
from tqdm import tqdm




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('WORDMAP_corpus.json', 'r') as j:
    word_map = json.load(j)


transformer = Transformer(d_model = config.d_model,
                            heads = config.heads,
                            num_layer = config.num_layers,
                            word_map = word_map)

transformer = transformer.to(device)

adam_custom = torch.optim.Adam(transformer.parameters(),
                                 lr = 0,
                                  betas = (0.9,0.98),
                                  eps = 1e-9)

trans_optim = AdamWarmup(model_size = config.d_model,
                        warmup_steps = 4000, 
                        optimizer = adam_custom)

criterion = LossWithLS(len(word_map), 0.1)








for epoch in range(config.epochs):

    sum_loss = 0

    for question, reply in tqdm(train_loader):

        batch_size = question.shape[0]

        question = question.to(device)
        reply = reply.to(device)

        reply_input = reply[:, :-1]
        reply_target = reply[:, 1:]

        question_mask, reply_input_mask, reply_target_mask = create_masks(question, reply_input, reply_target)

        out = transformer(question, question_mask, reply_input, reply_input_mask)
        loss = criterion(out, reply_target, reply_target_mask)
        trans_optim.optimizer.zero_grad()
        loss.backward()
        trans_optim.step()

        sum_loss += loss.item() * batch_size

        print(sum_loss)

    state = {'epoch': config.epochs, 'transformer': transformer, 'transformer_optimizer': trans_optim}
    torch.save(state, 'checkpoint_' + str(epoch) + '.pth.tar')


    

        
       



 
