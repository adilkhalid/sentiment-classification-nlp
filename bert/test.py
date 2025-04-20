import numpy as np

from bert.mini_BERT import MiniBERT
from bert.mini_BERT_tokenizer import MiniBERTTokenizer


def test_bert():
    # === Rebuild tokenizer and model (same config as training) ===
    tokenizer = MiniBERTTokenizer()
    tokenizer.build_vocab(["hello world", "my name is bert", "this is a test", "i love deep learning"])
    vocab_size = tokenizer.vocab_size
    max_len = 20
    embed_dim = 64
    ff_hidden_dim = 128
    num_layers = 2

    model = MiniBERT(vocab_size, max_len, embed_dim, ff_hidden_dim, num_layers)

    # === Input sentences ===
    sentence_a = "my name is [MASK]"
    sentence_b = "this is a test"

    tokens = tokenizer.tokenize(sentence_a, sentence_b)
    input_ids = tokens["input_ids"]
    token_type_ids = tokens["token_type_ids"]
    position_ids = tokens["position_ids"]
    tokens_list = tokens["tokens"]

    # === Forward pass ===
    mlm_logits, nsp_logits = model.forward(input_ids, token_type_ids, position_ids)

    # === MLM: Predict top token for each position ===
    print("Masked Language Predictions:")
    for i, token in enumerate(tokens_list):
        if token == "[MASK]":
            pred_id = int(np.argmax(mlm_logits[i]))
            predicted_token = tokenizer.index_to_word[pred_id]
            print(f"Position {i}: Predicted token: '{predicted_token}'")

    # === NSP: Prediction ===
    nsp_pred = np.argmax(nsp_logits)
    print("\nNSP Prediction:", "Is Next Sentence" if nsp_pred == 1 else "Not Next Sentence")


if __name__ == "__main__":
    test_bert()