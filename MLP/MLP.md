# Multi-Layer Perceptron (MLP) / Feedforward Neural Network (FNN)

## Why MLP/FNN for Sentiment Analysis?
MLPs (also called FNNs) are useful for text classification when **word order does not matter**. They:

1. **Treat text as a bag of words** – Unlike RNNs, MLPs do not process words sequentially.  
2. **Learn non-linear patterns** – By using multiple layers, they capture complex relationships in data.  
3. **Are efficient for short texts** – Since they do not need to remember past words, they train faster than RNNs.  

---

## How MLP Works for Sentiment Analysis

1. **Word Representation:**
   - Each word in a sentence is converted into a **numeric representation**.
   - This can be done using:
     - **One-hot encoding** (Sparse)
     - **TF-IDF** (Weighted)
     - **Pretrained Word Embeddings (Word2Vec, GloVe, FastText)** (Dense)

2. **Feature Aggregation:**
   - Since MLPs do not process text sequentially, **word vectors are combined into a single fixed-size representation**.
   - Common methods:
     - **Averaging word embeddings** (e.g., Word2Vec average)
     - **Concatenation**
     - **TF-IDF weighted sum**

3. **Feedforward Neural Network Layers:**
   - **Input Layer:** Receives the **aggregated word vector**.
   - **Hidden Layers:** Capture non-linear relationships in the data.
     \[
     h = \text{ReLU}(W_1 \cdot x + b_1)
     \]
   - **Output Layer:** Maps the hidden representation to a sentiment score.
     \[
     y_{\text{pred}} = \sigma(W_2 \cdot h + b_2)
     \]
   - `σ` is the **sigmoid function** (for binary classification).  

---

## Model Explanation (Trainable Embedding)

If using **trainable embeddings**, the embedding matrix is learned during training:

1. **Embedding Layer (Trainable)**
   - Converts each word into a dense vector of size `d` (embedding dimension).  
   - Shape: `(vocab_size, embedding_dim)`  
   - Optimized during training via **backpropagation**.

2. **Forward Pass (MLP with Trainable Embeddings)**
   - Each word embedding is **aggregated into a single vector** (e.g., via averaging).
   - The aggregated vector is passed through **fully connected layers**:
     \[
     h = \text{ReLU}(W_1 \cdot x + b_1)
     \]
     \[
     y_{\text{pred}} = \sigma(W_2 \cdot h + b_2)
     \]

3. **Prediction**
   - `y_pred > 0.5` → **Positive sentiment**
   - `y_pred <= 0.5` → **Negative sentiment**

---

## Model Explanation (Pretrained Word2Vec Embeddings)

When using **pretrained Word2Vec**, the word embeddings are **fixed** and not updated:

1. **Fixed Embedding Layer**
   - Words are represented using **precomputed** Word2Vec vectors.  
   - The embedding layer acts as a **lookup table**.  

2. **Feature Aggregation**
   - Common methods:
     - **Averaging embeddings** (simple and effective)
     - **Weighted sum (TF-IDF weighting)**

3. **MLP Classifier**
   - Takes the **aggregated word vector** and passes it through:
     - **Fully connected layers** with ReLU activation.
     - **Sigmoid activation** for final sentiment prediction.

---

## Differences Between MLP and RNN

| Feature | MLP / FNN | RNN |
|---------|----------|-----|
| **Processes words sequentially?** | ❌ No | ✅ Yes |
| **Understands word order?** | ❌ No | ✅ Yes |
| **Suitable for short texts?** | ✅ Yes | ❌ No (better for long texts) |
| **Handles long-term dependencies?** | ❌ No | ✅ Yes |
| **Training speed** | ✅ Faster | ❌ Slower |
| **Common embedding techniques** | One-hot, TF-IDF, Word2Vec (averaged) | Word2Vec, Trainable |

---

## Summary

- **MLPs treat text as a bag of words**, ignoring word order.  
- They work well for **short texts** and when **word order is not important**.  
- **Trainable embeddings** allow the model to learn task-specific word meanings.  
- **Pretrained embeddings** (e.g., Word2Vec) provide **rich word representations** without additional training.  
- Unlike RNNs, **MLPs do not maintain a hidden state**, making them less effective for long text sequences.

