import random

import numpy as np
import torch


def sample_sorted_alignment_minibatch_from_numpy_with_domain_y(domain_y, input_numpy, batch_size, sampleFlag, device):
    x_word, x_char, set_y = input_numpy
    ori_tags, sentence_length_Y, label_Y = set_y

    if sampleFlag == True:
        select_index = np.array(random.sample(range(len(x_word)), batch_size))
    else:
        select_index = np.array(range(len(x_word)))

    b_len = sentence_length_Y[select_index]
    maxL = np.max(b_len)

    idx = np.argsort(b_len)
    idx = idx[::-1]  # decreasing

    select_index = select_index[idx]  # Sort the mini batch

    b_x_word = x_word[select_index]
    b_x_char = x_char[select_index]

    ori_tags = np.array(ori_tags)  # ori_tags is a list
    b_ori_tags = ori_tags[select_index]
    b_sentence_length_Y = sentence_length_Y[select_index]
    b_type_Y = label_Y[select_index]

    cha_lengths = []
    for xxx in b_x_char:
        for xx in xxx:
            cha_lengths.append(len(xx))
    maxChar = np.max(cha_lengths)

    if maxChar < 5:
        maxChar = 5

    # TODO: padding words
    b_x_word_pad = []
    for aa in b_x_word:
        b_x_word_pad.append(np.pad(aa, (0, maxL - len(aa)), 'constant'))

    b_x_word_pad = np.array(b_x_word_pad)

    # TODO: padding character   'word 0' should be 'P_'
    b_x_char_flat_pad = []
    for ddd in b_x_char:
        for i in range(maxL):
            if i < len(ddd):
                dd = ddd[i]
            else:
                dd = np.zeros(1, dtype=np.int32)
            b_x_char_flat_pad.append(np.pad(dd, (0, maxChar - len(dd)), 'constant'))

    b_x_char_flat_pad = np.array(b_x_char_flat_pad)
    b_x_char_flat_pad = b_x_char_flat_pad.reshape(b_x_word_pad.shape[0], b_x_word_pad.shape[1], maxChar)

    # assert sum(b_sentence_length_Y) == b_x_char_flat_pad.shape[0]

    data_size = len(b_type_Y)
    maxL = len(b_type_Y[0])

    matrix_BIOES = np.full((data_size, maxL), 6)  # not only for 6, any number can be the placeholder.

    for i in range(data_size):  # loop batch size
        cur_line = b_type_Y[i]

        matrix_BIOES[i, 0:len(cur_line)] = cur_line

    # TODO: numpy to tensor
    b_x_word_pad = torch.from_numpy(b_x_word_pad.astype(np.int64)).to(device)
    b_x_char_flat_pad = torch.from_numpy(b_x_char_flat_pad.astype(np.int64)).to(device)
    b_sentence_length_Y = torch.from_numpy(b_sentence_length_Y.astype(np.int64)).to(device)

    b_domain_y = torch.from_numpy(np.array([domain_y] * batch_size).astype(np.int64)).to(device)

    minibatchInput = []
    minibatchInput.append(b_ori_tags)  # 0  numpy
    minibatchInput.append(b_x_word_pad)  # 1  tensor
    minibatchInput.append(b_x_char_flat_pad)  # 2  tensor
    minibatchInput.append(b_sentence_length_Y)  # 3  tensor

    minibatchInput.append(torch.from_numpy(matrix_BIOES).to(device))  # 9 tensor

    minibatchInput.append(b_domain_y)

    return minibatchInput


def sample_sorted_alignment_minibatch_from_numpy(input_numpy, batch_size, sampleFlag, device):
    x_word, x_char, set_y = input_numpy
    ori_tags, sentence_length_Y, label_Y = set_y

    if sampleFlag == True:
        select_index = np.array(random.sample(range(len(x_word)), batch_size))
    else:
        select_index = np.array(range(len(x_word)))

    b_len = sentence_length_Y[select_index]
    maxL = np.max(b_len)

    idx = np.argsort(b_len)
    idx = idx[::-1]  # decreasing

    select_index = select_index[idx]  # Sort the mini batch

    b_x_word = x_word[select_index]
    b_x_char = x_char[select_index]

    ori_tags = np.array(ori_tags)  # ori_tags is a list
    b_ori_tags = ori_tags[select_index]
    b_sentence_length_Y = sentence_length_Y[select_index]
    b_type_Y = label_Y[select_index]

    cha_lengths = []
    for xxx in b_x_char:
        for xx in xxx:
            cha_lengths.append(len(xx))
    maxChar = np.max(cha_lengths)

    if maxChar < 5:
        maxChar = 5

    # TODO: padding words
    b_x_word_pad = []
    for aa in b_x_word:
        b_x_word_pad.append(np.pad(aa, (0, maxL - len(aa)), 'constant'))

    b_x_word_pad = np.array(b_x_word_pad)

    # TODO: padding character   'word 0' should be 'P_'
    b_x_char_flat_pad = []
    for ddd in b_x_char:
        for i in range(maxL):
            if i < len(ddd):
                dd = ddd[i]
            else:
                dd = np.zeros(1, dtype=np.int32)
            b_x_char_flat_pad.append(np.pad(dd, (0, maxChar - len(dd)), 'constant'))

    b_x_char_flat_pad = np.array(b_x_char_flat_pad)
    b_x_char_flat_pad = b_x_char_flat_pad.reshape(b_x_word_pad.shape[0], b_x_word_pad.shape[1], maxChar)

    # assert sum(b_sentence_length_Y) == b_x_char_flat_pad.shape[0]

    data_size = len(b_type_Y)
    maxL = len(b_type_Y[0])

    matrix_BIOES = np.full((data_size, maxL), 6)  # not only for 6, any number can be the placeholder.

    for i in range(data_size):  # loop batch size
        cur_line = b_type_Y[i]

        matrix_BIOES[i, 0:len(cur_line)] = cur_line

    # TODO: numpy to tensor
    b_x_word_pad = torch.from_numpy(b_x_word_pad.astype(np.int64)).to(device)
    b_x_char_flat_pad = torch.from_numpy(b_x_char_flat_pad.astype(np.int64)).to(device)
    b_sentence_length_Y = torch.from_numpy(b_sentence_length_Y.astype(np.int64)).to(device)

    minibatchInput = []
    minibatchInput.append(b_ori_tags)  # 0  numpy
    minibatchInput.append(b_x_word_pad)  # 1  tensor
    minibatchInput.append(b_x_char_flat_pad)  # 2  tensor
    minibatchInput.append(b_sentence_length_Y)  # 3  tensor

    minibatchInput.append(torch.from_numpy(matrix_BIOES).to(device))  # 9 tensor

    return minibatchInput


def select_data_based_on_select_index(input_data, select_index):
    train_x_word, train_x_char, train_y = input_data
    ori_tags, sentence_length_Y, type_Y = train_y

    re_x_word = train_x_word[select_index]
    re_x_char = train_x_char[select_index]

    ori_tags = np.array(ori_tags)  # ori_tags is a list

    re_y = ori_tags[select_index], sentence_length_Y[select_index], type_Y[select_index]

    return [re_x_word, re_x_char, re_y]


def fix_nn(model, theta):
    def k_param_fn(tmp_model, name=None):
        if len(tmp_model._modules) != 0:
            for (k, v) in tmp_model._modules.items():
                if name is None:
                    k_param_fn(v, name=str(k))
                else:
                    k_param_fn(v, name=str(name + '.' + k))
        else:
            for (k, v) in tmp_model._parameters.items():
                if not isinstance(v, torch.Tensor):
                    continue
                tmp_model._parameters[k] = theta[str(name + '.' + k)]

    k_param_fn(model)
    return model
