# BERT Encoder (Mini) Decoder Only

## Architecture

<table>
    <tr>
        <th>Component</th>
        <th>Purpose</th>
    </tr>
    <tr>
        <td>Token Embeddings</td>
        <td>Convert each word into a vector using a lookup table</td>
    </tr>
    <tr>
        <td>Position Embedding</td>
        <td>Encode the order/position of each word in the sentence</td>
    </tr>
    <tr>
        <td>Segment Embeddings</td>
        <td>Indicate Whether a token belongs to sentence A or B</td>
    </tr>
    <tr>
        <td>Input Embedding</td>
        <td>Sum of token + position + sement embeddings</td>
    </tr>
    <tr>
        <td>Encoder Layers</td>
        <td>Stck of Transformer blocks (multi head attention + FFN + layer norm)</td>
    </tr>
    <tr>
        <td>[CLS] Token output</td>
        <td>The final representation of the [CLS] token is used for classification</td>
    </tr>
    <tr>
        <td>Masking</td>
        <td>Randomly mask some tokens for training (MLM)</td>
    </tr>
</table>

## BERT Diagram (High Level)
           Input
             ↓
     ┌─────────────────┐
     │ Self-Attention  │
     └─────────────────┘
             ↓
      Add & LayerNorm
             ↓
     ┌──────────────────────┐
     │ Feedforward + ReLU   │
     └──────────────────────┘
             ↓
      Add & LayerNorm
             ↓
           Output
