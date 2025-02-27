from typing import List


class IndexerFeatureExtractor:
    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def vocab_size(self):
        return len(self.objs_to_ints)

    def add_to_indexer(self, sentence: List[str]):
        for word in sorted(sentence):
            if word not in self.objs_to_ints:
                index = len(self.objs_to_ints)
                self.objs_to_ints[word] = index
                self.ints_to_objs[index] = word

    def extract_feature(self, sentence: List[str]):
        feature_vector = [0] * len(self.objs_to_ints)
        for word in sorted(sentence):
            idx = self.objs_to_ints.get(word, 0)
            if idx != -1:
                feature_vector[idx] = 1
        return feature_vector


