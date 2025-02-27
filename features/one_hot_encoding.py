# whitespace tokenization
from typing import List


class OneHotEncodingFeatureExtractor:
    # build vocabulary
    def __init__(self):
        self.vocabulary = set()

    def build_vocab(self, sentences: List[List[str]]):
        new_words = set(self.vocabulary)  # Keep existing vocabulary
        for sentence in sentences:
            for word in sentence:
                new_words.add(word)  # Add new words

        self.vocabulary = sorted(new_words)  # Keep order consistent

    def add_to_vocab(self, sentence: List[str]):
        """
        Adds words from a new sentence to the vocabulary dynamically.

        Args:
            sentence (List[str]): Tokenized words from a new sentence.
        """
        new_words = set(self.vocabulary)  # Keep existing vocabulary
        for word in sentence:
            new_words.add(word)  # Add new words

        self.vocabulary = sorted(new_words)  # Keep order consistent

    def extract_features(self, sentence: List[str]):
        processed_words = []
        for word in sorted(sentence):
            processed_words.append(word)

        feature_vector = [1 if word in processed_words else 0 for word in self.vocabulary]
        return feature_vector
