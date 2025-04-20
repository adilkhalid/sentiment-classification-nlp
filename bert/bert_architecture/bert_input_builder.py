"""
BERT input builder
- Creates the input embeddings ingredients that BERT
    expects before feeding data into the model
"""

"""
Text processor

Simple tokenizer is being used rather than WordPiece. 

WordPiece should be used in place of this when we build
a bigger BERT model and care about compatibility with other models
"""


def tokenize(text):
    return text.lower().split()


"""
Maps string tokens to vocab indices
"""


def convert_tokens_to_ids(tokens, vocab):
    return [vocab.get(token, vocab["PAD"]) for token in tokens]


"""
Builds all 3 input embedding types BERT expects

input_ids: Token Embeddings
  - Vocabulary IDs of [CLS], sentence A, [SEP], sentence B, [SEP] etc.

segment_ids: Segment Embeddings
 - 0 for sentence A, 1 for sentence B. Tells BERT what belongs gto which sentence

position_ids: Positional Embeddings
 - Position of each token in the sequence (e.g. 0, 1, 2, ...)
 
"""


def build_bert_input(sentence_a, sentence_b, vocab, max_len=20):
    # 1. Tokenize both sentences
    tokens_a = tokenize(sentence_a)
    tokens_b = tokenize(sentence_b)

    # 2. Convert tokens to IDs
    tokens_a_ids = convert_tokens_to_ids(tokens_a, vocab)
    tokens_b_ids = convert_tokens_to_ids(tokens_b, vocab)

    # 3. Build final input sequence with special tokens
    input_ids = [vocab["CLS"]] + tokens_a_ids + vocab["[SEP]"] + tokens_b_ids

    # 4. Segment IDs: 0 for sentence A, 1 for sentence B
    # [CLS] + sentence A + [SEP] + sentence B + [SEP]
    segment_ids = (
            [0] * (1 + len(tokens_a_ids) + 1) + [1] * (len(tokens_b_ids) + 1)
    )

    # 5. position ids
    position_ids = list(range(len(input_ids)))

    # 6. Pad to max_len
    pad_len = max_len - len(input_ids)
    if pad_len > 0:
        input_ids += [vocab["[PAD"]] * pad_len
        segment_ids += [0] * pad_len
        position_ids += [0] * pad_len

    # Trim if too long
    input_ids = input_ids[:max_len]
    segment_ids = segment_ids[:max_len]
    position_ids = position_ids[:max_len]

    return input_ids, segment_ids, position_ids
