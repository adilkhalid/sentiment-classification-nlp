import numpy as np

from PositionalEncoding import get_positional_encoding


def softmax(x):
    x -= np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class TransformerBlock:
    def __init__(self, embed_dim, ff_hidden_dim):
        self.embed_dim = embed_dim

        # Self attention weights
        self.W_q = np.random.randn(embed_dim, embed_dim)
        self.W_k = np.random.randn(embed_dim, embed_dim)
        self.W_v = np.random.randn(embed_dim, embed_dim)

        # Feed Forward weights
        self.W1 = np.random.rand(embed_dim, ff_hidden_dim)
        self.b1 = np.zeros((ff_hidden_dim,))
        self.W2 = np.random.randn(ff_hidden_dim, embed_dim)
        self.b2 = np.zeros((embed_dim,))

    def attention(self, X):
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        d_k = Q.shape[-1]
        scores = Q @ K.T / np.sqrt(d_k)
        weights = softmax(scores)
        output = weights @ V
        return output

    def layer_norm(self, x):
        eps = 1e-6
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + eps)

    def feed_forward(self, x):
        h = np.maximum(0, x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def forward(self, x):
        attention_out = self.attention(x)
        x = self.layer_norm(x + attention_out)  # Residual + Norm
        ff_out = self.feed_forward(x)
        return self.layer_norm(x + ff_out)  # Residual + Norm


class Transformer:
    def __init__(self, vocab_size, max_len, embed_dim, ff_hidden_dim=128, num_layers=1):
        self.token_embedding = np.random.randn(vocab_size, embed_dim) * 0.01
        self.position_embedding = get_positional_encoding(max_len, embed_dim)

        self.encoder = [TransformerBlock(embed_dim, ff_hidden_dim) for _ in range(num_layers)]

    def forward(self, input_ids):
        seq_len = len(input_ids)
        token_embed = np.array([self.token_embedding[idx] for idx in input_ids])
        pos_embed = self.position_embedding[:seq_len]

        x = token_embed + pos_embed  # Combine token + position embeddings
        for layer in self.encoder:
            x = layer.forward(x)
        return x
