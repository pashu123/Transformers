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



<h2 id="multi-head-attention">Multi Head Attention</h2>
<p>This gives the model the advantage of focusing on different words h ways (h is the number of heads). It broadens the model’s capability to focus on different positions and gives the attention layer multiple different representations.</p>

