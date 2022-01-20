import os
import pickle
from random import shuffle, sample
from typing import List
from collections import defaultdict
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


def generate_few_shot_dataset(file_path, sample_per_class, output_dir, file_name):
    """
    Less strict few-shot definition. file should be in IOB2 format.
    :param file_path:
    :param sample_per_class:
    :return:
    """
    class2sentences = defaultdict(list)
    sentence_buffer = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip().split()
            if len(line) < 2:  # line break
                if sentence_buffer:
                    related_classes = set([x[1][2:] for x in sentence_buffer if x[1] != 'O'])
                    for c in related_classes:
                        class2sentences[c].append(sentence_buffer)
                    sentence_buffer = []
            else:
                word, tag = line
                sentence_buffer.append([word, tag])

    # sample
    samples = []
    sample_quota = defaultdict(lambda: sample_per_class)
    sample_order = sorted(class2sentences.keys(), key=lambda x: len(class2sentences[x]), reverse=False)  # start sampling from small class
    print("Start Sampling")
    for c in sample_order:
        while sample_quota[c] > 0 and len(class2sentences[c]) > 0:  # keep sampling when still have quota for this class
            shuffle(class2sentences[c])
            s = class2sentences[c].pop()  # sample 1 sentence per time
            # check if it is valid sample
            classes_in_sentence = defaultdict(lambda: 0)
            for _, tag in s:
                if len(tag) > 2 and tag[:2] == "B-":  # to omit "O" class
                    tag = tag[2:]
                    classes_in_sentence[tag] += 1
            is_valid_sample = True
            for tag in classes_in_sentence.keys():
                new_quota = sample_quota[tag] - classes_in_sentence[tag]
                if new_quota < -sample_per_class:  # invalid, causing excessive samples in other class
                    is_valid_sample = False
                    break
            # add to sample pool
            if is_valid_sample:
                samples.append(s)
                for c, value in classes_in_sentence.items():
                    sample_quota[c] -= value

    # verbose the sample size
    print("Stop Sampling")
    class_size = defaultdict(lambda: 0)
    for s in samples:
        for _, tag in s:
            if len(tag) > 2 and tag[:2] == "B-":
                class_size[tag[2:]] += 1
    print(class_size)
    # generate file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, file_name)
    with open(output_path, "w+") as f:
        for s in samples:
            for word, tag in s:
                f.write("{}\t{}\n".format(word, tag))
            f.write("\n")
    print("{} is generated".format(output_path))


def sample_data(dataset, size_per_type: int):
    """
    fixme: sampling technique?
    :param dataset:
    :param size_per_type:
    :return:
    """
    tag2data = defaultdict(list)  # "ORG": [data, number of samples]
    tag2size = defaultdict(lambda: 0)
    for sentence, tags in dataset:
        related_types = get_type_occurrences(tags)
        for entity_type, occurrence in related_types.items():
            tag2data[entity_type].append([(sentence, tags), occurrence])
            tag2size[entity_type] += occurrence
    quota_for_types = {entity_type: size_per_type for entity_type in tag2data.keys()}
    sorted_keys = sorted(quota_for_types.keys(), key=lambda x: tag2size[x], reverse=False)  # ascending order
    selected_samples = []

    # find the appropriate samples
    for entity_type in sorted_keys:
        sample_pool = tag2data[entity_type][:]
        random.shuffle(sample_pool)
        while quota_for_types[entity_type] > 0 and len(
                sample_pool) > 0:  # when there hasn't been sufficient samples for this type
            # sample 1 sentence at a time
            (sentence, sentence_tag), _ = sample_pool.pop()
            # check validity
            related_types = get_type_occurrences(sentence_tag)
            is_valid_sample = True
            for t, o in related_types.items():
                if t == entity_type:
                    continue
                if quota_for_types[t] - o <= -size_per_type:  # todo: threshold = 2 * sample_size
                    is_valid_sample = False
                    break
            if is_valid_sample:
                selected_samples.append((sentence, sentence_tag))
                for t, o in related_types.items():
                    quota_for_types[t] -= o

    # print the sample data num
    tag2size = defaultdict(lambda: 0)
    for sentence, tag in selected_samples:
        related_types = get_type_occurrences(tag)
        for t, o in related_types.items():
            tag2size[t] += o
    for t, o in tag2size.items():
        print("{}={}".format(t, o))

    return selected_samples


def get_type_occurrences(tags):
    related_types = defaultdict(lambda: 0)  # "type": number of occurrence
    for word_tag in tags:
        if word_tag[:2] == "B-":
            related_types[word_tag[2:]] += 1
    return related_types


if __name__ == '__main__':
    # preprocess_word2vec("glove.6B.300d.txt")
    # split_dataset("data/wikigold", "data/wikigold/wikigold_iob.txt")
    generate_few_shot_dataset("data/bio_nlp_13/train.txt", 5, "data/bio_nlp_13_5", "train.txt")
