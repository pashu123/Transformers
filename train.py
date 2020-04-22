import torch
import torch.nn.functional as F
from dataset import train_loader,vocab_dict
from model import Transformer
import config
from tqdm import tqdm
from model import create_padding_mask, create_look_ahead_mask

# Getting the vocabulary size for the embedding matrix
vocab_size = len(vocab_dict)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## Setting up the transformer
transformer = Transformer(d_model = config.d_model,
                            heads = config.heads,
                            num_layers = config.num_layers,
                            vocab_size = vocab_size)


## Sending the transformer to device
transformer = transformer.to(device)


## Hack no. 1 setting the parameters of layer to xavier_uniform
for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)


## Want to train the loaded model
# checkpoint = torch.load('checkpoint.pth.tar')
# transformer = checkpoint['transformer']


## Hack no. 2 Got from pytorch transformer implementation
lr = 5.0 # learning rate
optimizer = torch.optim.SGD(transformer.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)



for epoch in range(config.epochs):

    tot_loss = 0
    count = 0
    enum = 0
    for (question, reply) in train_loader:

        batch_size = question.shape[0]

        src = question.to(device)
        target = reply.to(device)

        target_input = target[:, :-1]

        ## remember teacher forcing
        ys = target[:, 1:].contiguous().view(-1)

        src_mask = create_padding_mask(src)
        trg_mask = create_look_ahead_mask(target_input)

        preds = transformer(src, target_input, src_mask, trg_mask)

        preds = preds.view(-1, preds.size(-1))

        loss = F.cross_entropy(preds, ys, ignore_index = 0)

        optimizer.zero_grad()

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), 0.5)

        optimizer.step()

       	tot_loss += loss.item() * batch_size
        count += batch_size

        enum += 1

        if enum % 200 == 0:
            print("Loss: {:.3f}".format(tot_loss/count))
    
    print(f'{epoch} completed')

    state = {'epoch': config.epochs, 'transformer': transformer, 'transformer_optimizer': trans_optim}
    torch.save(state, 'checkpoint' + '.pth.tar')


 
