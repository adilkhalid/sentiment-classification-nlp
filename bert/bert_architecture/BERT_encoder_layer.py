import numpy as np

from bert.bert_architecture.self_attention import SelfAttention


class BERTEncoderLayer:
    """
    A simplified implementation of a single BERT encoder layer using NumPy.

    This class replicates the key components of the original Transformer encoder block:
        - Multi-head self-attention (assumed to be implemented in SelfAttention)
        - Add & Layer Normalization
        - Position-wise Feedforward Network
        - Residual Connections

    Note:
        This version assumes the existence of a `SelfAttention` class with a `.forward(x)` method
        that outputs attention-applied embeddings of the same shape as input `x`.

    Attributes:
        attention (SelfAttention): Self-attention module operating on input embeddings.
        W1, b1 (np.ndarray): Weights and bias for the first linear layer in the feedforward sublayer.
        W2, b2 (np.ndarray): Weights and bias for the second linear layer in the feedforward sublayer.
    """

    def __init__(self, embed_dim, ff_hidden_dim):
        """
        Initializes the BERT encoder layer with attention and feedforward components.

        Args:
            embed_dim (int): Dimensionality of input embeddings.
            ff_hidden_dim (int): Dimensionality of the hidden layer in the feedforward network.
        """
        self.attention = SelfAttention(embed_dim)

        # Feedforward network parameters
        self.W1 = np.random.randn(embed_dim, ff_hidden_dim) * 0.01
        self.b1 = np.zeros((ff_hidden_dim,))
        self.W2 = np.random.randn(ff_hidden_dim, embed_dim) * 0.01
        self.b2 = np.zeros((embed_dim,))

    def layer_norm(self, x, eps=1e-6):
        """
        Applies layer normalization over the last dimension of input.

        Args:
            x (np.ndarray): Input tensor to normalize.
            eps (float): Small constant to avoid division by zero.

        Returns:
            np.ndarray: Layer-normalized tensor.
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + eps)

    def feed_forward(self, x):
        """
        Applies a two-layer feedforward network with ReLU activation.

        Args:
            x (np.ndarray): Input tensor of shape (sequence_length, embed_dim).

        Returns:
            np.ndarray: Output tensor of shape (sequence_length, embed_dim).
        """
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU activation
        return h @ self.W2 + self.b2

    def forward(self, x):
        """
        Forward pass through the encoder layer.

        The sequence of operations is:
            1. Self-Attention with residual connection and layer normalization
            2. Feedforward network with residual connection and layer normalization

        Args:
            x (np.ndarray): Input embeddings of shape (sequence_length, embed_dim).

        Returns:
            np.ndarray: Output of the encoder layer (same shape as input).
        """
        # 1. Self-Attention + Residual + Norm
        attn_out = self.attention.forward(x)
        x = self.layer_norm(x + attn_out)

        # 2. Feedforward + Residual + Norm
        ff_out = self.feed_forward(x)
        x = self.layer_norm(x + ff_out)

        return x
