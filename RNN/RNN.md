# Recurrent Neural Network (RNN)

## Why RNN?
RNNs are useful for sequential data, especially in NLP tasks like sentiment analysis. They:

1. **Process text sequentially** – They remember previous words when predicting sentiment.  
2. **Maintain a hidden state** – Stores information about past words in a sentence.  
3. **Handle long texts better** – Unlike MLPs (which ignore context beyond a fixed window), RNNs learn dependencies over longer sequences.  

## RNN in Sentiment Analysis
1. Each word is fed **one at a time** into the network.  
2. The model **remembers previous words** through a hidden state.  
3. At the end of the sequence, the **final hidden state** is used to predict sentiment.  

---

## Model Explanation (with Trainable Embedding)
When using a trainable embedding layer, the model **learns word representations** during training instead of relying on precomputed embeddings like Word2Vec.

1. **Embedding Layer (Trainable)**
   - Converts each word into a dense vector representation (embedding).  
   - These embeddings are **learned** based on the dataset rather than being fixed.  
   - Shape: `(vocab_size, embedding_dim)`, where `embedding_dim` is the vector size for each word.  

2. **Weight Matrices:**
   - **W_xh (Input to Hidden):** Maps the word embedding to the hidden state.  
   - **W_hh (Hidden to Hidden):** Stores past word information, enabling the model to remember context.  
   - **W_hy (Hidden to Output):** Maps the hidden state to the sentiment prediction.

3. **Forward Pass:**
   - Each word `x_t` (represented by an embedding) is processed as:
     \[
     h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1})
     \]
   - The final hidden state is used for sentiment classification:
     \[
     y_{\text{pred}} = \sigma(W_{hy} \cdot h_T)
     \]
   - Here, `σ` is the **sigmoid function** (for binary classification).  

4. **Why Trainable Embeddings?**
   - They **adapt** to the dataset, improving accuracy.  
   - They allow the model to **capture task-specific nuances** in word meanings.  
   - They **update during training** via backpropagation and gradient descent.  

---

## Model Explanation (with Pretrained Word2Vec)
If using **pretrained Word2Vec embeddings**, the process is similar, but the embedding layer **does not update** during training. Instead, it acts as a fixed lookup table.

1. **W_xh (Input to Hidden):** Maps the **fixed** Word2Vec embedding to the hidden state.  
2. **W_hh (Hidden to Hidden):** Stores past word information and accumulates context.  
3. **W_hy (Hidden to Output):** Maps the hidden state to the sentiment prediction.  
4. **Final Prediction (`y_pred`):** Uses a **sigmoid activation** for binary classification.  

---

## Differences Between Trainable and Pretrained Embeddings
| Feature               | Trainable Embeddings     | Pretrained Word2Vec  |
|----------------------|-----------------------|----------------------|
| **Updates during training?** | ✅ Yes | ❌ No (fixed vectors) |
| **Captures dataset-specific nuances?** | ✅ Yes | ❌ No |
| **Requires large dataset?** | ❌ No (adapts to small data) | ✅ Yes (Word2Vec needs large corpus) |
| **Computational efficiency** | ❌ Slower (more parameters) | ✅ Faster (fixed embeddings) |

---

## Summary
- **Trainable embeddings** allow the model to learn task-specific word meanings.  
- **Pretrained Word2Vec embeddings** provide better generalization but are not updated during training.  
- Both approaches use **W_xh, W_hh, and W_hy** for processing sequential text and predicting sentiment.  

