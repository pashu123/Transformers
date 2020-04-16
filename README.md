# Transformers
Pytorch Implementation of Transformers Explained with Comments



<h1 id="introduction">Introduction</h1>
<p>The Transformer are based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. These models are superior in quality while being more parallelizable and requiring significantly less time to train. In this document we will describe the transformer model completely and finally make our transformer model in PyTorch and test it on Cornell Movie Dialogs Corpus to show some interesting result.</p>
.

<h1 id="features-of-transformers">Features of Transformers</h1>
<h2 id="not-sequential">Not Sequential</h2>



<h2 id="self-attention">Self Attention</h2>
<p>As the model processes each word (each position in the input sequence), self attention allows it to look at other positions in the input sequence for clues that can help lead to a better encoding for this word.</p>



<h2 id="multi-head-attention">Multi Head Attention</h2>
<p>This gives the model the advantage of focusing on different words h ways (h is the number of heads). It broadens the model’s capability to focus on different positions and gives the attention layer multiple different representations.</p>

<p><span id="fig:awesome_image1" label="fig:awesome_image1">[fig:awesome_image1]</span> <img src="6" title="fig:" alt="In another head ’heroes’ is attending to ’path’ and ’choose’" /></p>
<p><span id="fig:awesome_image2" label="fig:awesome_image2">[fig:awesome_image2]</span></p>