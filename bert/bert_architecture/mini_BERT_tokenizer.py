class MiniBERTTokenizer:
    """
       A simplified tokenizer inspired by the original BERT tokenizer.

       This tokenizer is designed for educational purposes and mimics the core behavior of BERT's preprocessing pipeline.
       It handles tokenization, vocabulary building, and the generation of token IDs, segment IDs, and position IDs.

       Key Differences from Original BERT Tokenizer:
           - Uses whitespace-based tokenization instead of WordPiece.
           - Simplified vocabulary building from raw corpus.
           - Handles only lowercase tokens and basic sentence pair formatting.

       Attributes:
           vocab (dict): Initial vocabulary with special tokens.
           word_to_index (dict): Mapping from words to their unique token IDs.
           index_to_word (dict): Mapping from token IDs to words.
           vocab_size (int): Size of the tokenizer's vocabulary.
       """
    def __init__(self):
        self.vocab = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3}
        self.word_to_index = dict(self.vocab)
        self.index_to_word = {v: k for k, v in self.word_to_index.items()}
        self.vocab_size = len(self.vocab)

    def build_vocab(self, corpus):
        """
        Builds a vocabulary from a given list of sentences.

        Adds new lowercase words found in the corpus to the tokenizer's vocabulary,
        assigning each a unique integer ID starting after the special tokens.

        Args:
            corpus (list of str): A list of raw text sentences.

        Example:
            tokenizer.build_vocab(["Hello world", "My name is BERT"])
        """
        for sentence in corpus:
            for word in sentence.lower().split():
                if word not in self.word_to_index:
                    idx = len(self.word_to_index)
                    self.word_to_index[word] = idx
                    self.index_to_word[idx] = word
        self.vocab_size = len(self.word_to_index)

    def tokenize(self, sentence_a, sentence_b=None):
        """
       Tokenizes one or two sentences into token IDs, segment IDs, and position IDs.

       The function:
           - Adds [CLS] at the beginning and [SEP] after each sentence.
           - Splits sentences into tokens (lowercased words).
           - Converts tokens to their corresponding IDs.
           - Assigns segment IDs (0 for sentence A, 1 for sentence B).
           - Computes position IDs for each token.

       Args:
           sentence_a (str): The first sentence (mandatory).
           sentence_b (str, optional): The second sentence (optional).

       Returns:
           dict: A dictionary with keys:
               - 'input_ids': List of token IDs.
               - 'token_type_ids': Segment IDs indicating sentence A or B.
               - 'position_ids': Position indices (0-based).
               - 'tokens': The list of tokens including special tokens.

       Example:
           tokenizer.tokenize("My name is BERT", "Nice to meet you")

           Output:
           {
               'input_ids': [1, 10, 11, 12, 13, 2, 14, 15, 16, 17, 2],
               'token_type_ids': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
               'position_ids': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
               'tokens': ['[CLS]', 'my', 'name', 'is', 'bert', '[SEP]', 'nice', 'to', 'meet', 'you', '[SEP]']
           }
       """
        tokens = ["[CLS]"] + sentence_a.lower().split() + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if sentence_b:
            tokens += sentence_b.lower().split() + ["[SEP]"]
            segment_ids += [1] * (len(sentence_b.lower().split()) + 1)  # +1 for the second [SEP]

        token_ids = [self.word_to_index.get(t, self.word_to_index["[PAD]"]) for t in tokens]
        position_ids = list(range(len(token_ids)))

        return {
            "input_ids": token_ids,
            "token_type_ids": segment_ids,
            "position_ids": position_ids,
            "tokens": tokens
        }
