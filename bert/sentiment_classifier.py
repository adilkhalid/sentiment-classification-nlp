import numpy as np


class SentimentClassifier:
    def __init__(self, embed_dim):
        self.W = np.random.randn(embed_dim, 1) * 0.01
        self.b = np.zeros(1)

    def forward(self, cls_embedding):
        logits = cls_embedding @ self.W + self.b
        return 1 / (1 + np.exp(-logits))  # Sigmoid
