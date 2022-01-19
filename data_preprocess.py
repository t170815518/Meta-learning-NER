import os
import pickle
from random import shuffle

import numpy as np

BUFFER_SIZE = 10000000


def preprocess_word2vec(emb_path):
    """
    Preprocess the download GloVe word vector file.
    :param emb_path: str, the path to the glove word embedding file

    ref:
        - https://stackoverflow.com/a/49389628/11180198
    """
    word2id = {}
    id2word = {}
    word_id = 0
    vectors = []

    with open(emb_path, 'rb') as f:
        line = f.readline()
        while line:
            fields = line.strip().split()
            word = fields[0].decode('utf-8')
            vector = fields[1:]
            word2id[word] = word_id
            id2word[word_id] = word
            vectors.append(vector)
            word_id += 1

            if word_id % 1000 == 0:
                print("{} words are processed".format(word_id))

            line = f.readline()

    matrix = np.array(vectors)
    np.save("pre_trained_embedding.npy", matrix)
    with open("word2id.pkl", "wb+") as f:
        pickle.dump(word2id, f)
    with open("id2word.pkl", "wb+") as f:
        pickle.dump(id2word, f)


def split_dataset(directory, file_path):
    sentences = []
    sentence = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            if len(line) < 2:
                sentences.append(sentence)
                sentence = []
            else:
                if len(line) == 2:
                    word, tag = line
                sentence.append([word, tag])
    shuffle(sentences)
    train_size = int(len(sentences) * 0.6)
    test_size = int(len(sentences) * 0.2)

    train_data = sentences[:train_size]
    test_data = sentences[train_size:train_size + test_size]
    val_data = sentences[train_size + test_size:]

    with open(os.path.join(directory, "train.txt"), "w+") as f:
        for sentence in train_data:
            for word, tag in sentence:
                f.write("{}\t{}\n".format(word, tag))
            f.write('\n')

    with open(os.path.join(directory, "test.txt"), "w+") as f:
        for sentence in test_data:
            for word, tag in sentence:
                f.write("{}\t{}\n".format(word, tag))
            f.write('\n')

    with open(os.path.join(directory, "valid.txt"), "w+") as f:
        for sentence in val_data:
            for word, tag in sentence:
                f.write("{}\t{}\n".format(word, tag))
            f.write('\n')


if __name__ == '__main__':
    # preprocess_word2vec("glove.6B.300d.txt")
    split_dataset("data/wikigold", "data/wikigold/wikigold.conll.txt")
