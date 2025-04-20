from typing import List

class IndexerFeatureExtractor:
    """
    A simple feature extractor that builds a vocabulary index from input text
    and converts sentences into binary feature vectors (bag-of-words style).

    This extractor:
        - Assigns a unique index to each word in the vocabulary.
        - Converts sentences into binary vectors indicating the presence of known words.
        - Ignores word frequency (i.e., a word is either present (1) or absent (0)).

    Attributes:
        objs_to_ints (dict): A mapping from words to unique integer indices.
    """

    def __init__(self):
        """
        Initializes an empty vocabulary indexer.
        """
        self.objs_to_ints = {}

    def vocab_size(self):
        """
        Returns the current size of the vocabulary.

        Returns:
            int: Number of unique words added to the indexer.
        """
        return len(self.objs_to_ints)

    def add_to_indexer(self, sentence: List[str]):
        """
        Adds words from a sentence to the vocabulary index.

        New words are added with unique indices. Sorting ensures deterministic indexing.

        Args:
            sentence (List[str]): A list of tokens (words) to add to the vocabulary.

        Example:
            add_to_indexer(["hello", "world"])
        """
        for word in sorted(sentence):
            if word not in self.objs_to_ints:
                index = len(self.objs_to_ints)
                self.objs_to_ints[word] = index

    def extract_feature(self, sentence: List[str]):
        """
        Converts a sentence into a binary feature vector.

        The vector has the same length as the vocabulary. A `1` at position `i` indicates
        that the word assigned to index `i` is present in the sentence.

        Args:
            sentence (List[str]): A list of tokens (words) to convert.

        Returns:
            List[int]: A binary feature vector of size `vocab_size()`.

        Example:
            Given vocabulary: {"hello": 0, "world": 1}
            extract_feature(["hello"]) returns [1, 0]
        """
        feature_vector = [0] * len(self.objs_to_ints)
        for word in sorted(sentence):
            idx = self.objs_to_ints.get(word, -1)
            if idx != -1:
                feature_vector[idx] = 1
        return feature_vector
