from typing import List

class NGramFeatureExtractor:
    """
    A feature extractor that converts tokenized sentences into n-gram bag-of-words feature vectors.

    This extractor builds a vocabulary of all observed n-grams and represents sentences as
    fixed-length vectors, where each dimension counts how often a specific n-gram appears.

    Supports unigrams (n=1), bigrams (n=2), trigrams (n=3), etc.

    Attributes:
        n (int): The size of the n-grams (e.g., 1 for unigrams, 2 for bigrams).
        ngram_to_index (dict): A mapping from n-gram tuples to unique integer indices.
        vocab_size (int): The total number of unique n-grams observed.
    """

    def __init__(self, n=1):
        """
        Initializes the NGramFeatureExtractor.

        Args:
            n (int): Size of the n-grams to extract (default is 1 for unigrams).
        """
        self.n = n
        self.ngram_to_index = {}
        self.vocab_size = 0

    def build_vocab(self, sentences: List[List[int]]):
        """
        Builds the n-gram vocabulary from a list of tokenized sentences.

        Args:
            sentences (List[List[int]]): A list of token ID sequences (e.g., [[1, 2, 3], [4, 5]]).
        """
        for sentence in sentences:
            ngrams = [tuple(sentence[i:i + self.n]) for i in range(len(sentence) - self.n + 1)]
            for ngram in ngrams:
                if ngram not in self.ngram_to_index:
                    self.ngram_to_index[ngram] = self.vocab_size
                    self.vocab_size += 1

    def add_to_vocab(self, sentence: List[int]):
        """
        Adds n-grams from a single sentence to the vocabulary.

        Args:
            sentence (List[int]): A tokenized sentence represented as a list of integers.
        """
        if len(sentence) < self.n:
            return
        ngrams = [tuple(sentence[i:i + self.n]) for i in range(len(sentence) - self.n + 1)]
        for ngram in ngrams:
            if ngram not in self.ngram_to_index:
                self.ngram_to_index[ngram] = self.vocab_size
                self.vocab_size += 1

    def extract_feature(self, sentence: List[int]):
        """
        Converts a sentence into a bag-of-ngrams feature vector.

        The output vector has a length equal to the size of the n-gram vocabulary,
        with each index representing the count of a specific n-gram in the sentence.

        Args:
            sentence (List[int]): A tokenized sentence represented as a list of integers.

        Returns:
            List[int]: A feature vector counting n-gram occurrences.
        """
        feature = [0] * self.vocab_size
        ngrams = [tuple(sentence[i:i + self.n]) for i in range(len(sentence) - self.n + 1)]

        for ngram in ngrams:
            if ngram in self.ngram_to_index:
                feature[self.ngram_to_index[ngram]] += 1  # Count occurrences
        return feature

    def model_details(self):
        """
        Returns the current model configuration and vocabulary mapping.

        Returns:
            dict: A dictionary containing:
                - 'ngram_to_index': The n-gram to index mapping (with tuple keys as strings).
                - 'n': The n-gram size.
                - 'vocab_size': Total number of unique n-grams.
        """
        return {
            "ngram_to_index": {str(k): v for k, v in self.ngram_to_index.items()},
            "n": self.n,
            "vocab_size": self.vocab_size
        }
