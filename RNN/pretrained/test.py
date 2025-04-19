import numpy as np
from data.dataloader import load_dataset
from utils.model_io import load_rnn_model, load_word2vec_model


def test_rnn():
    """Loads the trained RNN model and evaluates it on test dataset."""
    # Load the trained RNN model
    rnn = load_rnn_model("rnn_model")

    # Load the pretrained Word2Vec model (used during training)
    word2vec = load_word2vec_model("../../cbow/cbow_model")

    # Load test dataset
    dataset = load_dataset("../../dataset/test.csv")

    # Iterate through test dataset
    for sentence, label in dataset:
        # Convert words to Word2Vec vectors
        sentence_vectors = [word2vec.W1[word2vec.word_to_index.get(word, 0)] for word in sentence.split()]
        if not sentence_vectors:  # Skip empty sentences
            continue

        # Forward pass
        y_pred, _ = rnn.forward(sentence_vectors)

        # Convert probability to binary sentiment
        sentiment = "Positive" if y_pred >= 0.5 else "Negative"

        # Print results
        print(f"Sentence: {sentence}")
        print(f"Predicted Sentiment: {sentiment} | Actual Sentiment: {'Positive' if label == 1 else 'Negative'}\n")


if __name__ == "__main__":
    test_rnn()
