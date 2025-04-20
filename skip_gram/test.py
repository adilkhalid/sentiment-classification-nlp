import numpy as np

from data.dataloader import load_dataset
from skip_gram.SkipGram import SkipGram
from utils.model_io import load_word2vec_model


def test():
    word2vec = load_word2vec_model("skipgram")
    dataset = load_dataset('../dataset/test.csv')
    skipgram = SkipGram(word2vec)

    for sentence, label in dataset:
        words = sentence.split()

        target_word = words[0]
        target_idx = word2vec.word_to_index[target_word]

        y_pred = skipgram.forward(target_idx)

        top_indices = np.argsort(y_pred)[-3:][::-1]

        predicted_context_words = [word2vec.index_to_word[i] for i in top_indices]

        print(f"Sentence: {sentence}")
        print(f"Target Word: {target_word}")
        print(f"Predicted Context words: {predicted_context_words}\n")


if __name__ == "__main__":
    test()
