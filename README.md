# Transformers
Pytorch Implementation of Transformers Explained with Comments



<h1 id="introduction">Introduction</h1>
<p>The Transformer are based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. These models are superior in quality while being more parallelizable and requiring significantly less time to train. In this document we will describe the transformer model completely and finally make our transformer model in PyTorch and test it on Cornell Movie Dialogs Corpus to show some interesting result.</p>
.

<h1 id="features-of-transformers">Features of Transformers</h1>
<h2 id="not-sequential">Not Sequential</h2>
<br>
<br>

![1](https://user-images.githubusercontent.com/16246821/79481350-fc6ade80-802c-11ea-8f9f-4aa0591f23b6.png)

![2](https://user-images.githubusercontent.com/16246821/79481319-f1b04980-802c-11ea-9553-091795c73f4d.png)


<h2 id="self-attention">Self Attention</h2>
<p>As the model processes each word (each position in the input sequence), self attention allows it to look at other positions in the input sequence for clues that can help lead to a better encoding for this word.</p>

<br>

![4](https://user-images.githubusercontent.com/16246821/79481328-f4ab3a00-802c-11ea-9224-4024827fbb5e.png)




<h2 id="multi-head-attention">Multi Head Attention</h2>
<p>This gives the model the advantage of focusing on different words h ways (h is the number of heads). It broadens the model’s capability to focus on different positions and gives the attention layer multiple different representations.</p>

![5](https://user-images.githubusercontent.com/16246821/79481331-f5dc6700-802c-11ea-9df3-530615a44b54.png) ![6](https://user-images.githubusercontent.com/16246821/79481333-f674fd80-802c-11ea-9858-817f1729c042.png)



<h1 id="architecture">Architecture</h1>


![7](https://user-images.githubusercontent.com/16246821/79481335-f70d9400-802c-11ea-83f7-6f470fe46196.png)

<h2 id="input-embeddings">Input Embeddings</h2>
<p>First we encode every word into embedding vector i.e choose glove embedding, and since transformer accepts sentences so we define the Max Length which is no. of word embedding to be passed. Finally, we process the input in batches so a final tensor of <em>Embedding Dimension * Max Length * Batch Size</em> is processed.</p>

![8](https://user-images.githubusercontent.com/16246821/79481338-f8d75780-802c-11ea-8fea-eb179ce5ca52.png)

<h2 id="positional-encoding">Positional Encoding</h2>
<p>This technique is used because there is no notion of word order (1st word, 2nd word, ..) in the proposed architecture. All words of input sequence are fed to the network with no special order or position (unlike common RNN or ConvNet architectures), thus, model has no idea how the words are ordered. Consequently, a position-dependent signal is added to each word-embedding to help the model incorporate the order of words.</p>

![9](https://user-images.githubusercontent.com/16246821/79481339-f96fee00-802c-11ea-9470-ca511ec8a6cc.png)

![10](https://user-images.githubusercontent.com/16246821/79481341-faa11b00-802c-11ea-92e9-e6062725383d.png)


<h2 id="multi-head-attention-1">Multi-Head Attention</h2>
<p>The General Framework of Attention is given by</p>
<p>Attention(Q,K,V) = Softmax(Q <span class="math inline"><em>K</em><sup><em>T</em></sup></span> / <span class="math inline"><em>d</em><sub><em>h</em></sub></span>)V</p>
<p>where Q is Query Vector, K is Key Vector and V is Value vector.</p>

![11](https://user-images.githubusercontent.com/16246821/79481342-fb39b180-802c-11ea-88dd-0d639396987b.png)

<p>In case of Multi-Head attention we have, For each head i: <span class="math inline"><em>h</em><em>e</em><em>a</em><em>d</em><sub><em>i</em></sub></span> = Attention(<span class="math inline"><em>Q</em><em>W</em><sub><em>i</em></sub><sup><em>Q</em></sup></span>, <span class="math inline"><em>K</em><em>W</em><sub><em>i</em></sub><sup><em>K</em></sup></span>, <span class="math inline"><em>V</em><em>W</em><sub><em>i</em></sub><sup><em>V</em></sup></span>)</p>
<p>Finally all the attention head is concatenated and is passed through linear layer of same size as input so that the dimensions do not alter. We computed ’h’ different attention heads. Concatenation of heads is not enough to transfer information between heads and so the concatenated heads are passed through the linear layer.</p>
