from typing import List


class OneHotEncodingFeatureExtractor:
    """
    A basic one-hot encoding feature extractor for whitespace-tokenized text.

    This extractor:
        - Builds a vocabulary of unique words across input sentences.
        - Represents each input sentence as a binary vector where each position corresponds to a word in the vocabulary.
        - The feature vector has a 1 if the word is present in the sentence, 0 otherwise.
        - Sorting is used to ensure vocabulary consistency and determinism.

    Attributes:
        vocabulary (List[str]): A sorted list of unique words used for encoding.
    """

    def __init__(self):
        """
        Initializes an empty vocabulary.
        """
        self.vocabulary = set()

    def build_vocab(self, sentences: List[List[str]]):
        """
        Builds the vocabulary from a list of tokenized sentences.

        Each word from all input sentences is added to the vocabulary.
        The vocabulary is sorted to ensure consistent feature ordering.

        Args:
            sentences (List[List[str]]): A list of tokenized sentences.
        """
        new_words = set(self.vocabulary)
        for sentence in sentences:
            for word in sentence:
                new_words.add(word)
        self.vocabulary = sorted(new_words)

    def add_to_vocab(self, sentence: List[str]):
        """
        Adds new words from a single sentence to the vocabulary dynamically.

        Useful when the vocabulary is built incrementally.

        Args:
            sentence (List[str]): A list of words from a new sentence.
        """
        new_words = set(self.vocabulary)
        for word in sentence:
            new_words.add(word)
        self.vocabulary = sorted(new_words)

    def extract_features(self, sentence: List[str]):
        """
        Extracts a one-hot encoded feature vector for the input sentence.

        Each vector element corresponds to a word in the vocabulary and is set to:
            - 1 if the word is present in the input sentence
            - 0 otherwise

        Args:
            sentence (List[str]): A list of words from the input sentence.

        Returns:
            List[int]: A binary feature vector of length equal to the vocabulary size.
        """
        processed_words = sorted(sentence)
        feature_vector = [1 if word in processed_words else 0 for word in self.vocabulary]
        return feature_vector
