# Long Short-Term Memory (LSTM)

## Why LSTM?
LSTMs are an improvement over standard RNNs because they **better handle long-term dependencies**. They:

1. **Solve the vanishing gradient problem** – Unlike vanilla RNNs, LSTMs use a **gating mechanism** to retain information over long sequences.  
2. **Remember important words** – They decide which words to **keep or forget** using gates.  
3. **Work well for long texts** – They perform better than regular RNNs on longer sentences and documents.  

---

## How LSTM Works for Sentiment Analysis

1. **Each word is fed sequentially into the network.**  
2. **The LSTM cell maintains a hidden state and a memory cell.**  
3. **Gates regulate the flow of information:**
   - **Forget Gate** – Decides what to discard.  
   - **Input Gate** – Decides what new information to add.  
   - **Output Gate** – Controls what the next hidden state should be.  
4. **Final hidden state is used to predict sentiment.**  

---

## LSTM Cell Structure

An LSTM unit consists of:

- **Forget Gate** ($f_t$): Controls what past information to remove.  
- **Input Gate** ($i_t$): Decides how much new information to add.  
- **Cell State** ($C_t$): Stores long-term memory.  
- **Output Gate** ($o_t$): Determines what gets passed to the next time step.  

Mathematically, at time step $t$:

1. **Forget Gate**  
   $$
   f_t = \sigma(W_f \cd
] + b_i)
   $$

3. **Candidate Cell State** (New memory content)  
   $$
   \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
   $$

4. **Update Cell State** (Final memory update)  
   $$
   C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
   $$

5. **Output Gate**  
   $$
   o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
   $$

6. **Hidden State Update**  
   $$
   h_t = o_t \odot \tanh(C_t)
   $$

Here,  
- $\sigma$ is the **sigmoid activation function**, which outputs values between 0 and 1.  
- $\tanh$ squashes values between -1 and 1.  
- $\odot$ denotes element-wise multiplication.  

---

## Model Explanation (with Trainable Embedding)

If using **trainable embeddings**, the embedding matrix is **learned** during training.

1. **Embedding Layer (Trainable)**
   - Converts words into dense vectors of size $d$ (embedding dimension).  
   - Shape: $(\text{vocab\_size}, \text{embedding\_dim})$  
   - Optimized through **backpropagation**.

2. **LSTM Forward Pass**
   - Each word’s embedding $x_t$ is passed into the LSTM unit.  
   - The hidden state and memory cell update according to the **LSTM equations** above.

3. **Final Sentiment Prediction**
   - The last hidden state $h_T$ is passed to a fully connected layer with **sigmoid activation**:
     $$
     y_{\text{pred}} = \sigma(W_{hy} \cdot h_T)
     $$

---

## Model Explanation (with Pretrained Word2Vec Embeddings)

Instead of **learning embeddings**, we use **pretrained Word2Vec vectors**.

1. **Fixed Embedding Layer**
   - Uses precomputed Word2Vec embeddings.  
   - These embeddings **do not update** during training.  

2. **LSTM Forward Pass**
   - Each word’s embedding $x_t$ is passed into the LSTM, updating the hidden state $h_t$.  

3. **Final Prediction**
   - The last hidden state $h_T$ is used for classification:
     $$
     y_{\text{pred}} = \sigma(W_{hy} \cdot h_T)
     $$

---

## Differences Between LSTM and RNN

| Feature               | RNN | LSTM |
|-----------------------|----|------|
| **Handles long-term dependencies?** | ❌ No | ✅ Yes |
| **Uses memory cell?** | ❌ No | ✅ Yes |
| **Solves vanishing gradient problem?** | ❌ No | ✅ Yes |
| **Better for long texts?** | ❌ No | ✅ Yes |
| **Training speed** | ✅ Faster | ❌ Slower |
| **Common embedding techniques** | Word2Vec, Trainable | Word2Vec, Trainable |

---

## Summary

- **LSTMs improve RNNs** by adding a **memory cell** to capture long-term dependencies.  
- **Trainable embeddings** allow the model to **learn task-specific word representations**.  
- **Pretrained Word2Vec embeddings** provide **rich word representations** without additional training.  
- Unlike regular RNNs, **LSTMs use gates to control information flow**, solving the vanishing gradient problem.  

