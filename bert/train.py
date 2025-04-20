import numpy as np

from bert.bert_architecture.mini_BERT_tokenizer import MiniBERTTokenizer
from bert.mini_BERT import MiniBERT
from dataset.dataloader import load_dataset
from utils.functions import binary_cross_entropy, sigmoid
from utils.model_io import save_sentiment_bert


def train():
    # === Hyperparameters ===
    embed_dim = 64
    ff_hidden_dim = 128
    num_layers = 2
    max_len = 20
    epochs = 5
    lr = 0.01

    # === Load Dataset ===
    dataset = load_dataset("../../dataset/train.csv")

    # === Tokenizer and Model ===
    tokenizer = MiniBERTTokenizer()
    tokenizer.build_vocab([s for s, _ in dataset])
    vocab_size = tokenizer.vocab_size

    model = MiniBERT(vocab_size, max_len, embed_dim, ff_hidden_dim, num_layers)

    # === Add sentiment classification head ===
    embed_dim = 64  # or whatever you used in MiniBERT
    sentiment_weights = np.random.randn(embed_dim, 1)  # ✅ correct shape: (64, 1)
    sentiment_bias = np.zeros(1)

    for epoch in range(epochs):
        total_loss = 0
        for sentence, label in dataset:
            tokens = tokenizer.tokenize(sentence)

            input_ids = tokens["input_ids"]
            token_type_ids = tokens["token_type_ids"]
            position_ids = tokens["position_ids"]

            # === Forward Pass ===
            # Get the [CLS] embedding from the encoder output
            encoder_output = model.forward(input_ids, token_type_ids, position_ids)
            cls_embedding = encoder_output[0]  # Shape: (embed_dim,)

            logits = cls_embedding @ sentiment_weights + sentiment_bias

            y_pred = sigmoid(logits)

            # === Loss + Backprop ===
            loss = binary_cross_entropy(label, y_pred)
            total_loss += loss

            grad = y_pred - label  # dL/dlogits

            # Update sentiment head weights
            sentiment_weights -= lr * np.outer(cls_embedding, grad)
            sentiment_bias -= lr * grad

        print(f"Epoch {epoch + 1}: Sentiment Loss = {total_loss:.4f}")

    # Save model
    save_sentiment_bert(model, tokenizer, sentiment_weights, sentiment_bias)


if __name__ == "__main__":
    train()
