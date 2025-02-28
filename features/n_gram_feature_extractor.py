from typing import List


class NGramFeatureExtractor:

    def __init__(self, n=1):
        self.n = n
        self.ngram_to_index = {}
        self.vocab_size = 0

    def build_vocab(self, sentences: List[List[int]]):
        """Builds vocabulary of n-grams from a list of sentences."""
        for sentence in sentences:
            ngrams = [tuple(sentence[i:i + self.n]) for i in range(len(sentence) - self.n + 1)]
            for ngram in ngrams:
                if ngram not in self.ngram_to_index:
                    self.ngram_to_index[ngram] = self.vocab_size
                    self.vocab_size += 1

    def add_to_vocab(self, sentence: List[int]):
        if len(sentence) < self.n:
            return
        ngrams = [tuple(sentence[i:i + self.n]) for i in range(len(sentence) - self.n + 1)]
        for ngram in ngrams:
            if ngram not in self.ngram_to_index:
                self.ngram_to_index[ngram] = self.vocab_size
                self.vocab_size += 1

    def extract_feature(self, sentence: List[int]):
        """Converts a sentence into an N-Gram Bag of Words feature vector."""
        feature = [0] * self.vocab_size
        ngrams = [tuple(sentence[i:i + self.n]) for i in range(len(sentence) - self.n + 1)]

        for ngram in ngrams:
            if ngram in self.ngram_to_index:
                feature[self.ngram_to_index[ngram]] += 1  # Count occurrences
        return feature
