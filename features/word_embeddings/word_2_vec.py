import numpy as np
from typing import Tuple, List


class Word2Vec:
    def __init__(self, window_size=2, embedding_dim=50):
        self.window_size = window_size  # Context window size
        self.embedding_dim = embedding_dim  # Size of word embeddings
        self.word_to_index = {}  # Mapping: word -> index
        self.index_to_word = {}  # Mapping: index -> word
        self.vocab_size = 0  # Number of unique words
        self.W1 = None  # Word embedding matrix
        self.W2 = None  # Output layer for word prediction

    def build_vocabulary(self, sentences: List[List[str]]):
        """
        Builds a vocabulary index from a list of tokenized sentences.
        """
        unique_words = set(word for sentence in sentences for word in sentence)
        self.vocab_size = len(unique_words)
        self.word_to_index = {word: i for i, word in enumerate(unique_words)}
        self.index_to_word = {i: word for word, i in self.word_to_index.items()}

        # Initialize word embeddings (random values)
        self.W1 = np.random.uniform(-0.8, 0.8, (self.vocab_size, self.embedding_dim))
        self.W2 = np.random.uniform(-0.8, 0.8, (self.embedding_dim, self.vocab_size))

    def generate_cbow_training_data(self, sentences: List[List[str]]) -> List[Tuple[List[int], int]]:
        """
        Generates training data for CBOW and Skip-gram.
        - mode="cbow": context words → target word
        - mode="skipgram": target word → context words
        """
        training_data = []

        for sentence in sentences:
            for i in range(self.window_size, len(sentence) - self.window_size):
                context = sentence[i - self.window_size:i] + sentence[i + 1:i + self.window_size + 1]
                target = sentence[i]

                # Ensure words exist in the vocabulary
                if target not in self.word_to_index:
                    continue

                target_index = self.word_to_index[target]
                context_indices = [self.word_to_index[word] for word in context if word in self.word_to_index]

                if len(context_indices) > 0:
                    training_data.append((context_indices, target_index))

        return training_data

    def generate_skipgram_training_data(self, sentences: List[List[str]], window_size: int) -> List[Tuple[int, int]]:
        """
        Generates training data for Skip-gram.

        Args:
            sentences (List[List[str]]): Tokenized sentences.
            window_size (int): Number of words to consider as context.

        Returns:
            List[Tuple[int, int]]: (target_word, context_word)
        """
        training_data = []

        for sentence in sentences:
            for i in range(len(sentence)):
                target = sentence[i]

                if target not in self.word_to_index:
                    continue  # Skip unknown words

                target_index = self.word_to_index[target]

                start = max(0, i - window_size)
                end = min(len(sentence), i + window_size + 1)

                for j in range(start, end):
                    if i != j and sentence[j] in self.word_to_index:
                        training_data.append((target_index, self.word_to_index[sentence[j]]))

        return training_data

    def sentence_to_vector(self, sentence: List[str]):
        vectors = [self.W1[self.word_to_index[word]] for word in sentence]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.embedding_dim)
