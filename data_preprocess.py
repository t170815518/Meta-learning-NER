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
    glove_words = np.loadtxt(emb_path, dtype='str')
    words = glove_words[:, 0]
    vectors = glove_words[:, 1:].astype('float')
    np.save("word_vectors.npy", vectors)

    with open('word_list.pkl', 'wb+') as f:
        words = words.tolist()
        pickle.dump(words, f)


if __name__ == '__main__':
    preprocess_word2vec("glove.6B.300d.txt")
