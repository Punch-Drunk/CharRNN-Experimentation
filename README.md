# Character-Level Text Generation with LSTMs

This project explores **character-level recurrent neural networks (Char-RNNs)** for natural language modeling and text generation.  
Unlike word-level models, Char-RNNs learn directly from raw characters, capturing spelling, punctuation, and stylistic patterns without tokenization.

---

## 📂 Datasets
Two corpora were obtained from [Project Gutenberg](https://www.gutenberg.org/):
- **War and Peace** by Leo Tolstoy (~3.2M characters, 105 unique chars)
- **Collected Essays of John Stuart Mill** (~3.1M characters, 137 unique chars)

Preprocessing steps:
- Removed licensing/footnotes
- Built `char2idx` and `idx2char` dictionaries
- Created sequences of length 100 with offset targets for next-character prediction

---

## 🧠 Model Architecture
Implemented in PyTorch:
- **Embedding Layer**: maps characters to dense vectors
- **LSTM Layers**: 2–8 layers, hidden sizes 128–512, dropout=0.3
- **Output Layer**: fully connected → log-softmax over vocabulary

---

## ⚙️ Training
- **Loss**: Negative Log Likelihood (NLLLoss)  
- **Optimizer**: Adam (lr=7e-4) + ReduceLROnPlateau scheduler  
- **Mixed Precision**: CUDA AMP for efficiency  
- **Batch Sizes**: 256–1024 depending on configuration  

Progress monitored with `tqdm`, and text samples generated each epoch using multinomial sampling with temperature scaling.

---

## 🧪 Experiments & Results
**Tolstoy – High Capacity (4L, 512H, 256E, 20 epochs)**  
- Loss ↓ from 4.65 → 0.87  
- Generated coherent narrative prose with dialogue and names  
- Example: `"I am learning to speak! I will tell you, my dear boy—I don’t understand that something was in somebody!" said Nicholas.`

**Tolstoy – Reduced Capacity (2L, 128H)**  
- Final loss: 2.61  
- Faster training but degraded grammar and consistency  

**J.S. Mill – High Capacity**  
- Final loss: 0.55 in just 5 epochs  
- Generated structured philosophical arguments with domain vocabulary  
- Example: `"My opinion on cats is a true belief that Germany remain under the study of equality..."`

**J.S. Mill – Deep Narrow (8L, 256H)**  
- Loss: 1.14  
- Stable training, but long-range coherence weaker than wide model  

**Key Findings**  
- **Model width > depth** for Char-RNN performance  
- Philosophical text easier to model than narrative fiction  
- Larger models capture long-range dependencies more effectively  

---

## 🚀 Future Work
- Compare with Transformer-based char models  
- Explore hybrid word+char embeddings  
- Test different sequence lengths and sampling strategies  
- Scale to larger datasets (poetry, code, multi-language corpora)

---

## 📎 References
- [Andrej Karpathy – The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)  
- [char-rnn (Torch)](https://github.com/karpathy/char-rnn)  
- [Project Gutenberg](https://www.gutenberg.org/)
