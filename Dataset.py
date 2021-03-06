#!/usr/bin/python
# -*- coding: UTF-8 -*-
import logging
import os
import pathlib
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
import random


# todo: remove the magic number
CORPUS_SIZE = 400000
ALPHABET_SIZE = 269


class Dataset:
    """The data wrapper. One dataset class per domain."""
    domain2id = {}
    id2domain = {}
    total_train_sample_size = 0
    # used for char-level features
    char2id = {}
    id2char = {}
    encountered_chars = set()

    def __init__(self, domain_name: str, word2id: {}, id2word: {}, device, char2id: {} = None,
                 is_update_char2id: bool = True):
        self.is_update_char2id = is_update_char2id
        self.device = device
        self.id2word = id2word
        self.word2id = word2id
        self.name: str = domain_name
        if char2id:
            Dataset.char2id = char2id
            Dataset.id2char = {index: char for char, index in char2id.items()}
            Dataset.encountered_chars = set(char2id.keys())

        self.id2tag = {}
        self.tag2id = {}
        self.tag_num = 0

        # attribute place holder
        self.train = None
        self.test = None
        self.valid = None

        self.read_data()

        # update the class attribute
        domain_id = len(Dataset.domain2id)
        Dataset.domain2id[domain_name] = domain_id
        Dataset.id2domain[domain_id] = domain_name

    def process_data(self, data_batch):
        """
        process the raw sentence into PyTorch Tensor
        :return: input_feature, true_tags, domain_tag
        """
        batch_size = len(data_batch)

        word_seq = [torch.LongTensor(x[0]) for x in data_batch]  # index vector should be LongTensor
        word_seq = pad_sequence(word_seq, batch_first=True, padding_value=CORPUS_SIZE)

        # pad the seq tensor into [BATCH_SIZE, MAX_SEQ_LEN, MAX_CHAR_LEN] (batch_first)
        char_seq = [x[1] for x in data_batch]
        max_seq_len = max([len(x) for x in char_seq])
        max_char_len = max([len(x) for seq in char_seq for x in seq])
        padded_char_seq = torch.ones([batch_size, max_seq_len, max_char_len], dtype=torch.long) * ALPHABET_SIZE
        #   fill the padded_char_seq
        for i, sentence in enumerate(char_seq):
            for j, word in enumerate(sentence):
                for z, char in enumerate(word):
                    padded_char_seq[i, j, z] = char

        tag_seq = [torch.LongTensor(x[2]) for x in data_batch]
        tag_seq = pad_sequence(tag_seq, batch_first=True, padding_value=-1)

        domain_tag = torch.LongTensor([Dataset.domain2id[self.name]] * len(data_batch))
        # send to cuda
        word_seq = word_seq.to(self.device)
        padded_char_seq = padded_char_seq.to(self.device)
        tag_seq = tag_seq.to(self.device)
        domain_tag = domain_tag.to(self.device)
        if padded_char_seq.size()[-1] < 5:
            logging.info("there is batch with char_seq less than 5")

        return (word_seq, padded_char_seq), tag_seq, domain_tag

    def read_data(self):
        """
        reads the data and convert to numeric values e.g.,
            [[[WORD_ID_SEQ], [CHAR_SEQ],[TAG_SEQUENCE]], [[WORD_ID_SEQ], [CHAR_SEQ],[TAG_SEQUENCE]], ...]
        get the id2tag and tag2id
        """
        train_path = os.path.join("data", self.name, "train.txt")
        test_path = os.path.join("data", self.name, "test.txt")
        valid_path = os.path.join("data", self.name, "valid.txt")
        self.train = self.__read_file(train_path)
        self.test = self.__read_file(test_path)
        self.valid = self.__read_file(valid_path)
        # update class attributes
        self.tag_num = len(self.tag2id)
        Dataset.total_train_sample_size += len(self.train)

    def __read_file(self, file_path):
        data = []
        word_seq = []
        char_seq = []
        tag_seq = []
        cache_path = str(file_path[:-3]) + "pkl"
        # is_cached = pathlib.Path(cache_path).exists()
        is_cached = False

        if is_cached:
            logging.info("Loading {}".format(cache_path))
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
        else:
            logging.info("Loading {}".format(file_path))
            with open(file_path, 'r') as f:  # encoding='utf-8-sig' for wnut17
                for line in f:
                    line = line.strip().split()
                    if len(line) < 1:  # line-break
                        if word_seq != []:  # to avoid empty samples
                            data.append([word_seq, char_seq, tag_seq])
                        word_seq = []
                        tag_seq = []
                        char_seq = []
                    else:
                        if len(line) == 4:  # CONLL2003
                            word, _, _, tag = line
                        elif len(line) == 2:  # others
                            word, tag = line
                        else:
                            raise SyntaxError("Unexpected line: {} (at {})".format(line, file_path))
                        # convert to id
                        if word.lower() not in self.word2id.keys():
                            word_id = CORPUS_SIZE
                        else:
                            word_id = self.word2id[word.lower()]

                        if tag not in self.tag2id.keys():
                            new_id = len(self.tag2id)
                            self.tag2id[tag] = new_id
                            self.id2tag[new_id] = tag
                        tag = self.tag2id[tag]

                        chars = []
                        for char in word:
                            if self.is_update_char2id:
                                if char not in Dataset.encountered_chars:
                                    new_id = len(Dataset.char2id)
                                    Dataset.char2id[char] = new_id
                                    Dataset.id2char[new_id] = char
                                    Dataset.encountered_chars.add(char)
                                chars.append(self.char2id[char])
                            else:  # for test domain
                                if char not in Dataset.encountered_chars:
                                    chars.append(ALPHABET_SIZE)  # omit the new char
                                else:
                                    chars.append(self.char2id[char])

                        word_seq.append(word_id)
                        tag_seq.append(tag)
                        char_seq.append(chars)
            with open(cache_path, 'wb+') as f:
                pickle.dump(data, f)
                logging.info("the data is cached to {}".format(cache_path))

        return data

    def iter_train(self, batch_size):
        random.shuffle(self.train)

        for start_id in range(0, len(self.train), batch_size):
            end_id = min(len(self.train), start_id + batch_size)
            selected_data = self.train[start_id: end_id]
            (word_seq, char_seq), true_tags, domain_tag = self.process_data(selected_data)
            yield (word_seq, char_seq), true_tags, domain_tag
