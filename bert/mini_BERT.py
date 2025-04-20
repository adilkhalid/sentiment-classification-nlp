import numpy as np

from bert.bert_architecture.BERT_embedding import BERTEmbedding
from bert.bert_architecture.BERT_encoder import BERTEncoder
from bert.bert_architecture.masked_language_model_head import MaskedLanguageModelHead
from bert.bert_architecture.next_sentence_prediction_head import NextSentencePredictionHead
from bert.sentiment_classifier import SentimentClassifier


class MiniBERT:
    def __init__(self, vocab_size, max_len, embed_dim=64, ff_hidden_dim=128, num_layers=2):
        self.embedding = BERTEmbedding(vocab_size, max_len, embed_dim)
        self.encoder = BERTEncoder(num_layers, embed_dim, ff_hidden_dim)
        self.mlm_head = MaskedLanguageModelHead(embed_dim, vocab_size)
        self.nsp_head = NextSentencePredictionHead(embed_dim)
        self.embed_dim = embed_dim
        self.sentiment_head = SentimentClassifier(embed_dim)

    def forward(self, input_ids, token_type_ids, position_ids):
        """
        Forward pass through the entire BERT architecture.
        Returns:
            - mlm_predictions: shape (seq_len, vocab_size)
            - nsp_prediction: shape (2,)
        """
        embeddings = self.embedding.forward(input_ids, token_type_ids, position_ids)
        encoded = self.encoder.forward(embeddings)

        mlm_logits = self.mlm_head.forward(encoded)  # (seq_len, vocab_size)
        nsp_logits = self.nsp_head.forward(encoded[0])  # Only use [CLS] token

        sentiment_logits = self.sentiment_head.forward(embeddings)

        return mlm_logits, nsp_logits, sentiment_logits
