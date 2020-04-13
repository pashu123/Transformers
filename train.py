import json
import torch
import torch.nn.functional as F
from dataset import train_loader,vocab_dict
from model import Transformer
from extra_optimization import *
import config
from tqdm import tqdm
from model import create_padding_mask, create_look_ahead_mask


vocab_size = len(vocab_dict)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



transformer = Transformer(d_model = config.d_model,
                            heads = config.heads,
                            num_layers = config.num_layers,
                            vocab_size = vocab_size)

transformer = transformer.to(device)


adam_custom = torch.optim.Adam(transformer.parameters(),
                                 lr = 0,
                                  betas = (0.9,0.98),
                                  eps = 1e-9)

trans_optim = AdamWarmup(model_size = config.d_model,
                        warmup_steps = 4000, 
                        optimizer = adam_custom)








for epoch in range(config.epochs):

    tot_loss = 0
    count = 0
    enum = 0
    for (question, reply) in tqdm(train_loader):

        batch_size = question.shape[0]

        src = question.to(device)
        target = reply.to(device)

        target_input = target[:, :-1]
        
        ## remember teacher forcing
        ys = target[:, 1:].contiguous().view(-1)

        src_mask = create_padding_mask(src)
        trg_mask = create_look_ahead_mask(target_input)

        preds = transformer(src, target_input, src_mask, trg_mask)

        print(preds.shape)        
        preds = preds.view(-1, preds.size(-1))

        loss = F.cross_entropy(preds, ys, ignore_index = 0)
    
        print(loss)
        print(loss.shape)

        trans_optim.optimizer.zero_grad()

        loss.backward()

        trans_optim.step()

       	tot_loss += loss.item() * batch_size
        count += batch_size

        enum += 1

        if enum % 10 == 0:
            print("Loss: {:.3f}".format(tot_loss/count))


    state = {'epoch': config.epochs, 'transformer': transformer, 'transformer_optimizer': trans_optim}
    torch.save(state, 'checkpoint_' + str(epoch) + '.pth.tar')


    

        
       



 
