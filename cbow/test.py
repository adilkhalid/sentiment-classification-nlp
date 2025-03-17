import numpy as np

from cbow import CBOW
from data.dataloader import load_dataset
from utils.model_io import load_word2vec_model


def test():
    word2vec = load_word2vec_model()
    dataset = load_dataset('../data/test.csv')
    cbow = CBOW(word2vec)

    for sentence, label in dataset:
        words = sentence.split()

        if len(words) < 3:
            continue
        target_idx = len(words) // 2
        target_word = words[target_idx]

        context_words = words[:target_idx] + words[target_idx + 1:]
        context_indices = [word2vec.word_to_index[word] for word in context_words if word in word2vec.word_to_index]

        y_pred = cbow.forward(context_indices)

        predicted_index = np.argmax(y_pred)
        predicted_word = word2vec.index_to_word[predicted_index]

        print(f"Sentence: {sentence}")
        print(f"Actual Missing Word: {target_word}")
        print(f"Predicted Word: {predicted_word}\n")


if __name__ == "__main__":
    test()
