import numpy as np
import pickle

from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from gensim.scripts.word2vec2tensor import word2vec2tensor
from gensim.scripts.glove2word2vec import glove2word2vec


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


if __name__ == '__main__':
    preprocess_word2vec("glove.6B.300d.txt")
