# Neural Network Hacks Tried
`Xavier Initialization` : All layers of the transformers initialized with xavier uniform. [Xavier Uniform](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)<br>
`Gradient Clipping`: Gradient clipping to avoid exploding gradient problem. [Gradient Clipping](https://arxiv.org/pdf/1211.5063.pdf)<br>
`SGD with optimizer`: Got from official pytorch implemenation of transformers. [SGD optimizer and scheduler](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)<br>
# Hacks to Try
`Adam Optimizer with scheduler`: As mentioned in the transformers paper. [Transformers](https://arxiv.org/abs/1706.03762) <br>
`Beam Search with length normalization`: Beam search avoids neural text Degeneration. [Beam Search](https://arxiv.org/abs/1809.00069) <br>
`Avoid Neural Degenaration with Nucleus Sampling`: Nucleus Sampling works better than Beam Search. [Nucleus Sampling](https://arxiv.org/abs/1904.09751) <br>
# Transformers
Pytorch Implementation of Transformers Explained with Comments

<h1 id="introduction">Introduction</h1>
<p>The Transformer are based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. These models are superior in quality while being more parallelizable and requiring significantly less time to train. In this document we will describe the transformer model completely and finally make our transformer model in PyTorch and test it on Cornell Movie Dialogs Corpus to show some interesting result.</p>
.

<h1 id="features-of-transformers">Features of Transformers</h1>
<h2 id="not-sequential">Not Sequential</h2>
<br>
<br>


<p align="center">
  <img width="600" height="150" src="https://user-images.githubusercontent.com/16246821/79481350-fc6ade80-802c-11ea-8f9f-4aa0591f23b6.png">
</p>

<p align="center">
  <img width="500" height="125" src="https://user-images.githubusercontent.com/16246821/79481319-f1b04980-802c-11ea-9553-091795c73f4d.png">
</p>

```
The whole input is fed into transformer at once, whereas for sequential models like rnns, one at a time.
```


<h2 id="self-attention">Self Attention</h2>
<p>As the model processes each word (each position in the input sequence), self attention allows it to look at other positions in the input sequence for clues that can help lead to a better encoding for this word.</p>

<br>

<p align="center">
  <img width="450" height="100" src="https://user-images.githubusercontent.com/16246821/79481328-f4ab3a00-802c-11ea-9224-4024827fbb5e.png">
</p>

```
There is a high correlation between 'man' and 'battle' and 'man' and 'struggle' which is captured by self attention.
```




<h2 id="multi-head-attention">Multi Head Attention</h2>
<p>This gives the model the advantage of focusing on different words h ways (h is the number of heads). It broadens the model’s capability to focus on different positions and gives the attention layer multiple different representations.</p>


<table>
  <tr>
    <td><img width="100" height="300" src="https://user-images.githubusercontent.com/16246821/79481331-f5dc6700-802c-11ea-9df3-530615a44b54.png">
  </td>
    <td><img width="100" height="300" src="https://user-images.githubusercontent.com/16246821/79481333-f674fd80-802c-11ea-9858-817f1729c042.png"></td>
  </tr>
 </table>

```
In one head 'heroes' is attending to 'powers' and 'graced'
```
```
In another head 'heroes' is attending to 'path' and 'choose'
```



<h1 id="architecture">Architecture</h1>


<p align="center">
  <img width="625" height="400" src="https://user-images.githubusercontent.com/16246821/79481335-f70d9400-802c-11ea-83f7-6f470fe46196.png">
</p>

```
The full model architecture of the transformer. (Image source: Fig 1 & 2 in Vaswani, et al., 2017.)
```

<h2 id="input-embeddings">Input Embeddings</h2>
<p>First we encode every word into embedding vector i.e choose glove embedding, and since transformer accepts sentences so we define the Max Length which is no. of word embedding to be passed. Finally, we process the input in batches so a final tensor of <em>Embedding Dimension * Max Length * Batch Size</em> is processed.</p>

```
The input to the transformer is embedding dimension times Max length and we give batches of those.
```

<p align="center">
  <img width="225" height="230" src="https://user-images.githubusercontent.com/16246821/79481338-f8d75780-802c-11ea-8fea-eb179ce5ca52.png">
</p>

<h2 id="positional-encoding">Positional Encoding</h2>
<p>This technique is used because there is no notion of word order (1st word, 2nd word, ..) in the proposed architecture. All words of input sequence are fed to the network with no special order or position (unlike common RNN or ConvNet architectures), thus, model has no idea how the words are ordered. Consequently, a position-dependent signal is added to each word-embedding to help the model incorporate the order of words.</p>


<p align="center">
  <img width="600" height="150" src="https://user-images.githubusercontent.com/16246821/79481339-f96fee00-802c-11ea-9470-ca511ec8a6cc.png">
</p>

<p align="center">
  <img width="500" height="125" src="https://user-images.githubusercontent.com/16246821/79481341-faa11b00-802c-11ea-92e9-e6062725383d.png">
</p>

```
A real example of positional encoding with a toy embedding size of 4 (The Illustrated Transformer by Jay Allamar)
```

<h2 id="multi-head-attention-1">Multi-Head Attention</h2>
<p>The General Framework of Attention is given by</p>
<p>Attention(Q,K,V) = Softmax(Q <span class="math inline"><em>K</em><sup><em>T</em></sup></span> / <span class="math inline"><em>d</em><sub><em>h</em></sub></span>)V</p>
<p>where Q is Query Vector, K is Key Vector and V is Value vector.</p>

<p align="center">
  <img width="500" height="170" src="https://user-images.githubusercontent.com/16246821/79481342-fb39b180-802c-11ea-88dd-0d639396987b.png">
</p>

```
Here d_h is embedding size/h  and h is no. of attention heads.
```

<p>In case of Multi-Head attention we have, For each head i: <span class="math inline"><em>h</em><em>e</em><em>a</em><em>d</em><sub><em>i</em></sub></span> = Attention(<span class="math inline"><em>Q</em><em>W</em><sub><em>i</em></sub><sup><em>Q</em></sup></span>, <span class="math inline"><em>K</em><em>W</em><sub><em>i</em></sub><sup><em>K</em></sup></span>, <span class="math inline"><em>V</em><em>W</em><sub><em>i</em></sub><sup><em>V</em></sup></span>)</p>
<p>Finally all the attention head is concatenated and is passed through linear layer of same size as input so that the dimensions do not alter. We computed ’h’ different attention heads. Concatenation of heads is not enough to transfer information between heads and so the concatenated heads are passed through the linear layer.</p>

<h2 id="residual-learning">Residual Learning</h2>
<p>We are learning what’s left of (residual), without learning a new representation. You are learning the ’remaining’ only. If the block doesn’t learn anything, then your F(X) would be 0, and that it what makes the training go much faster, since learning a completely new representation is omitted. Therefor , the model can default to using the identity function if the layer is not beneficial.</p>
<p><strong>Either learn something useful, or don’t learn anything!</strong></p>

<p align="center">
  <img width="200" height="300" src="https://user-images.githubusercontent.com/16246821/79481345-fbd24800-802c-11ea-8ffd-af7d8d10fc06.png">
</p>

<h2 id="layer-normalization">Layer Normalization</h2>
<p>In order to prevent the values of the outputs from becoming bigger. We have performed a lot of operations which may cause the values of the layer output to become bigger.So we use Layer Norm to normalize them back again.</p>

<h2 id="masked-multi-head-attention">Masked Multi-Head Attention</h2>
<p>For self-attention, we don’t want our decoder to attend to future word. Otherwise, the model will cheat and learn to look at future words. At testing time, we don’t have future words! We are predicting one word at a time(running the decoder for a number of timesteps, just like an LSTM at testing time). So this will be incompatible during testing(inference). Therefore, the decoder is only allowed to attend to earlier positions. During testing time, it can only attend to what has been generated so far. So we need to resemble the testing time scenario during training as well.</p>

### Install
This project requires **Python** and the following Python libraries installed:

- [PyTorch](https://pytorch.org/)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included.

### Code
`dataset.py`: Reads the implemented tokenized dataset. (`Cornell Movie dialog Corpus`)<br>
`model.py`: Generic implementation of pytorch transformers.<br>
`train.py`: Training Loop<br>
`config.py`: Configuration of the model<br>
`chat.py`: Loads the model and allows interactive chatting on terminal.<br>

