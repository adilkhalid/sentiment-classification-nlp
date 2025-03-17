import math
from collections import defaultdict
from typing import List


class TFIDF:
    def __init__(self):
        self.idf = {}

    def compute_idf(self, documents: List[List[str]]):
        doc_size = len(documents)
        word_document_count = defaultdict(int)

        for doc in documents:
            unique_words = set(doc)
            for word in unique_words:
                word_document_count[word] += 1

        for word, doc_count in word_document_count.items():
            self.idf[word] = math.log(doc_size / (doc_count + 1))
        return self.idf

    def compute_tfidf(self, document: List[str]):
        """
        Computes the TF-IDF score for each word in a document.

        Args:
            document (List[str]): A tokenized list of words.

        Returns:
            dict: A dictionary mapping each word to its TF-IDF score.
        """

        word_count = defaultdict(int)
        total_words = len(document)

        for word in document:
            word_count[word] += 1
        tf_values = {}
        for word, count in word_count.items():
            tf_values[word] = count / total_words

        tfidf_values = {word: tf_values[word] * self.idf.get(word, 0) for word in
                        document}  # Default IDF = 0 if not in corpus
        return tfidf_values
