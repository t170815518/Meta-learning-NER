import os
import pickle
from random import shuffle
from typing import List

import numpy as np
import torch


BUFFER_SIZE = 10000000
PROCESSED_WORD2VEC_TENSOR_PATH = "pre_trained.pt"


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

    matrix = np.array(vectors).astype(float)
    print(matrix.shape)
    tensor = torch.Tensor(matrix)
    torch.save(tensor, PROCESSED_WORD2VEC_TENSOR_PATH)
    print("matrix is saved to {}".format(PROCESSED_WORD2VEC_TENSOR_PATH))
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
    # convert the tag to IOB2
    for i in range(len(sentences)):
        sentence = sentences[i]
        words = [x[0] for x in sentence]
        tags = [x[1] for x in sentence]
        iob2(tags)
        sentences[i] = [[x, y] for x, y in zip(words, tags)]

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


def iob2(tags: List[str]):
    """
    Ref: https://gist.github.com/allanj/b9bd448dc9b70d71eb7c2b6dd33fe4ef
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


if __name__ == '__main__':
    # preprocess_word2vec("glove.6B.300d.txt")
    split_dataset("data/wikigold", "data/wikigold/wikigold_iob.txt")
