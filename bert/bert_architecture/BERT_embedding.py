import numpy as np


class BERTEmbedding:
    """
    A simplified BERT-style embedding layer implemented using NumPy.

    This class mimics the embedding mechanism of the original BERT model by combining:
        - Token embeddings (word-level)
        - Segment embeddings (to distinguish between sentence A and sentence B)
        - Position embeddings (to encode the position of each token in the sequence)

    All embeddings are initialized with small random values (standard normal scaled by 0.01),
    and combined via element-wise addition.

    Attributes:
        embed_dim (int): Dimensionality of each embedding vector.
        token_embedding (np.ndarray): Embedding matrix for tokens of shape (vocab_size, embed_dim).
        segment_embedding (np.ndarray): Embedding matrix for segments (0 or 1), shape (2, embed_dim).
        position_embedding (np.ndarray): Embedding matrix for positions, shape (max_len, embed_dim).
    """

    def __init__(self, vocab_size, max_len, embed_dim):
        """
        Initializes the embedding matrices for tokens, segments, and positions.

        Args:
            vocab_size (int): Total number of tokens in the vocabulary.
            max_len (int): Maximum length of input sequences.
            embed_dim (int): Dimensionality of the embeddings.
        """
        self.embed_dim = embed_dim
        self.token_embedding = np.random.randn(vocab_size, embed_dim) * 0.01
        self.segment_embedding = np.random.randn(2, embed_dim) * 0.01
        self.position_embedding = np.random.randn(max_len, embed_dim) * 0.01

    def forward(self, token_ids, segment_ids, position_ids):
        """
        Computes the combined embeddings for a sequence of tokens.

        For each position `i` in the input sequence, it retrieves:
            - The token embedding for `token_ids[i]`
            - The segment embedding for `segment_ids[i]` (0 for Sentence A, 1 for Sentence B)
            - The positional embedding for `position_ids[i]`

        It then adds these three vectors to form a final embedding vector for each position.

        Args:
            token_ids (List[int]): List of token indices.
            segment_ids (List[int]): List of segment identifiers (0 or 1).
            position_ids (List[int]): List of position indices (usually range(len(token_ids))).

        Returns:
            np.ndarray: Combined embeddings of shape (sequence_length, embed_dim)
        """
        embeddings = []
        for i in range(len(token_ids)):
            token_vec = self.token_embedding[token_ids[i]]
            segment_vec = self.segment_embedding[segment_ids[i]]
            position_vec = self.position_embedding[position_ids[i]]
            combined = token_vec + segment_vec + position_vec
            embeddings.append(combined)
        return np.array(embeddings)
