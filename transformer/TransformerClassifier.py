import numpy as np


class TransformerClassifier:
    def __init__(self, transformer, embed_dim):
        self.transformer = transformer
        self.output_weights = np.random.randn(embed_dim, 1)
        self.output_bias = np.zeros(1)

    def forward(self, input_ids):
        x = self.transformer.forward(input_ids)
        pooled = np.mean(x, axis=0)  # Average over sequence
        logits = pooled @ self.output_weights + self.output_bias
        y_pred = 1 / (1 + np.exp(-logits))  # Sigmoid
        return y_pred, pooled