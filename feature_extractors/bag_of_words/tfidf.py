import math
from collections import defaultdict
from typing import List


class TFIDF:
    """
    A simple Term Frequency–Inverse Document Frequency (TF-IDF) calculator.

    TF-IDF is a numerical statistic that reflects how important a word is to a document
    in a collection or corpus. This class computes IDF values from a corpus and allows
    calculating TF-IDF vectors for new documents.

    Attributes:
        idf (dict): A mapping from words to their computed inverse document frequency (IDF) scores.
    """

    def __init__(self):
        """
        Initializes an empty TF-IDF model with no IDF values.
        """
        self.idf = {}

    def compute_idf(self, documents: List[List[str]]):
        """
        Computes the Inverse Document Frequency (IDF) values from a list of tokenized documents.

        The IDF score for a word is computed as:
            IDF(word) = log(N / (df + 1))
        where:
            - N is the total number of documents
            - df is the number of documents that contain the word
            - 1 is added to avoid division by zero

        Args:
            documents (List[List[str]]): A list of documents, where each document is a list of tokens (words).

        Returns:
            dict: A dictionary mapping each word to its IDF value.
        """
        doc_size = len(documents)
        word_document_count = defaultdict(int)

        for doc in documents:
            unique_words = set(doc)
            for word in unique_words:
                word_document_count[word] += 1

        for word, doc_count in word_document_count.items():
            self.idf[word] = math.log(doc_size / (doc_count + 1))  # Add 1 to avoid div by 0
        return self.idf

    def compute_tfidf(self, document: List[str]):
        """
        Computes the TF-IDF score for each word in a single document using precomputed IDF values.

        Term Frequency (TF) is calculated as:
            TF(word) = (Number of times the word appears in the document) / (Total words in the document)

        TF-IDF is then:
            TF-IDF(word) = TF(word) * IDF(word)

        If a word does not have a corresponding IDF value (i.e., wasn't in the training corpus),
        it is assigned an IDF of 0 by default.

        Args:
            document (List[str]): A tokenized document (list of words).

        Returns:
            dict: A dictionary mapping each word in the document to its TF-IDF score.
        """
        word_count = defaultdict(int)
        total_words = len(document)

        for word in document:
            word_count[word] += 1

        tf_values = {word: count / total_words for word, count in word_count.items()}

        tfidf_values = {
            word: tf_values[word] * self.idf.get(word, 0)
            for word in document
        }

        return tfidf_values
