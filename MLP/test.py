from data.dataloader import load_dataset
from utils.model_io import load_word2vec_model, load_mlp_model


def test():
    word2vec = load_word2vec_model("mlp")
    mlp = load_mlp_model()
    dataset = load_dataset('../dataset/test.csv')

    correct_predictions = 0
    total_samples = len(dataset)

    for sentence, label in dataset:
        sentence_vector = word2vec.sentence_to_vector(sentence.lower().split())

        y_pred, hidden_output = mlp.forward(sentence_vector)
        sentiment = 1 if y_pred >= 0.5 else 0  # Convert probability to binary

        if sentiment == label:
            correct_predictions += 1  # Track correct predictions

        print(f"Sentence: {sentence}")
        print(f"True Sentiment: {'Positive' if label == 1 else 'Negative'}")
        print(f"Predicted Sentiment: {'Positive' if sentiment == 1 else 'Negative'}\n")

    # Compute and print accuracy
    accuracy = correct_predictions / total_samples
    print(f"Model Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    test()
