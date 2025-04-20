import numpy as np

from utils.functions import softmax


class NextSentencePredictionHead:
    def __init__(self, embed_dim):
        self.W = np.random.randn(embed_dim, 2) * 0.01
        self.b = np.zeros(2)

    def forward(self, cls_embedding):
        logits = cls_embedding @ self.W + self.b  # (2,)
        return softmax(logits)
