import numpy as np
from data.dataloader import load_dataset
from utils.model_io import load_lstm_model, load_word2vec_model
from lstm import LSTM


def test_lstm():
    # Load pretrained components
    word2vec = load_word2vec_model("../cbow/cbow_model")
    lstm = load_lstm_model("lstm_model")

    # Load test dataset
    dataset = load_dataset("../dataset/test.csv")

    correct = 0
    total = 0

    for sentence, label in dataset:
        sentence_vectors = [
            word2vec.W1[word2vec.word_to_index[word]]
            for word in sentence.lower().split()
            if word in word2vec.word_to_index
        ]

        if not sentence_vectors:
            continue

        y_pred, _, _ = lstm.forward(sentence_vectors)
        prediction = int(y_pred >= 0.5)

        print(f"Sentence: {sentence}")
        print(
            f"Predicted: {'Positive' if prediction == 1 else 'Negative'} | Actual: {'Positive' if label == 1 else 'Negative'}\n")

        if prediction == label:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    test_lstm()
