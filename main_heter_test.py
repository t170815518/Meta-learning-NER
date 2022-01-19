import argparse
import copy
import os
import pickle

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from nn_model.crflayer import MyCRF
from nn_model.seq_feature_extractor import SeqFeatureExtractor
from utils.BIOES_ENCODER import get_TAG_and_pattens
from utils.functions import sample_sorted_alignment_minibatch_from_numpy, select_data_based_on_select_index
from utils.nermetric import get_metric_number_with_RE, get_ner_seg_score_from_listmatrix


def get_target_input_data_test(testPath):
    test_x_word = pickle.load(open(os.path.join(testPath, 'test_x_word_mapping'), 'rb'))
    test_x_char = pickle.load(open(os.path.join(testPath, 'test_x_char_mapping'), 'rb'))
    test_y = pickle.load(open(os.path.join(testPath, 'test_y.pickle'), 'rb'))

    return [test_x_word, test_x_char, test_y]


def get_target_input_data_train(testPath, datarate):
    test_x_word = pickle.load(open(os.path.join(testPath, 'train_x_word_mapping'), 'rb'))
    test_x_char = pickle.load(open(os.path.join(testPath, 'train_x_char_mapping'), 'rb'))
    test_y = pickle.load(open(os.path.join(testPath, 'train_y.pickle'), 'rb'))

    ori_tags, sentence_length_Y, label_Y = test_y

    N = len(test_x_word)

    select_index = list(range(0, int(N * datarate)))

    re_x_word = test_x_word[select_index]
    re_x_char = test_x_char[select_index]

    ori_tags = np.array(ori_tags)  # ori_tags is a list

    re_y = ori_tags[select_index], sentence_length_Y[select_index], label_Y[select_index]

    return [re_x_word, re_x_char, re_y]


def get_target_input_data_dev(testPath):
    test_x_word = pickle.load(open(os.path.join(testPath, 'dev_x_word_mapping'), 'rb'))
    test_x_char = pickle.load(open(os.path.join(testPath, 'dev_x_char_mapping'), 'rb'))
    test_y = pickle.load(open(os.path.join(testPath, 'dev_y.pickle'), 'rb'))

    return [test_x_word, test_x_char, test_y]


def check_without_fine_tuning(batch_size, feature_extractor, crf_layer, check_data, device):
    feature_extractor.eval()
    crf_layer.eval()

    print('check performance')
    # =============split data ===============
    data_L = len(check_data[0])

    need_loop = int(np.ceil(data_L / batch_size))

    metric_matrix = []

    for lp in range(need_loop):
        print('check %d of %d' % (lp, need_loop))
        startN = lp * batch_size
        endN = (lp + 1) * batch_size
        if endN > data_L:
            endN = data_L

        select_index = np.array(list(range(startN, endN)))

        select_data = select_data_based_on_select_index(check_data, select_index)

        inputTensors = sample_sorted_alignment_minibatch_from_numpy(select_data, batch_size, sampleFlag=False,
                                                                    device=device)

        encoder_features, combined_h_features = feature_extractor(inputTensors)
        scores, paths = crf_layer.viterbi_decode(encoder_features, inputTensors)

        g_seq = inputTensors[4]
        lens = inputTensors[3]
        batch_metric_number = get_metric_number_with_RE(g_seq, paths, lens, RE_PATTENS)

        metric_matrix.extend(batch_metric_number)

    print('check performance size:', len(metric_matrix))
    Precison, Recall, F1 = get_ner_seg_score_from_listmatrix(metric_matrix)

    print("%.2f" % (Precison * 100))
    print("%.2f" % (Recall * 100))
    print("%.2f" % (F1 * 100))
    return Precison, Recall, F1


def check_with_fine_tuning(taskname, ratio, batch_size, best_feature, best_crf, feature_extractor, crf_layer,
                           train_check_data, dev_check_data, test_check_data, device):
    no_epoch = 50

    opt_all = optim.Adam([{'params': feature_extractor.wordrep.word_embed.parameters(),
                           'lr': 1e-4, 'weight_decay': 0},
                          {'params': feature_extractor.base_params},
                          {'params': crf_layer.parameters()}],
                         lr=1e-3, weight_decay=1e-6)

    num_each_epoch = int(np.round(len(train_check_data[0]) / (batch_size)))
    total_iter = num_each_epoch * no_epoch

    check_point_iter_no = int(num_each_epoch / 2) - 2
    if check_point_iter_no == 0:
        check_point_iter_no = 1

    best_dev_F1 = 0

    for iter in range(total_iter):

        cur_epoch = int(np.ceil(iter / num_each_epoch + 0.00000001))
        cur_iter = iter % num_each_epoch
        print("epoch:%d/%d,iteration:%d/%d" % (cur_epoch, no_epoch, cur_iter, num_each_epoch))

        inputTensors = sample_sorted_alignment_minibatch_from_numpy(train_check_data, batch_size, sampleFlag=True,
                                                                    device=device)

        feature_extractor.train()
        crf_layer.train()

        encoder_features, combined_h_features = feature_extractor(inputTensors)
        crf_loss = crf_layer.crf_neglog_loss(encoder_features, inputTensors)

        opt_all.zero_grad()
        crf_loss.backward()

        clip_grad_norm_(list(feature_extractor.wordrep.word_embed.parameters()) +
                        list(feature_extractor.base_params) +
                        list(crf_layer.parameters()), 5)
        opt_all.step()

        # check performance
        if iter % check_point_iter_no == 0 and iter != 0:

            Precison, Recall, F1 = check_without_fine_tuning(batch_size, feature_extractor, crf_layer, test_check_data,
                                                             device)

            # print('dev F1:',F1)

            if best_dev_F1 < F1:
                best_dev_F1 = F1
                best_feature.load_state_dict(copy.deepcopy(feature_extractor.state_dict()))  # pickle.loads(
                # pickle.dumps(feature_extractor))   #deep copy?
                best_crf.load_state_dict(copy.deepcopy(crf_layer.state_dict()))  # pickle.loads(pickle.dumps(crf_layer))

            # Precison, Recall, F1 = check_without_fine_tuning(batch_size, feature_extractor, crf_layer,
            # test_check_data, device)

            with open('/MetaNER/output/heter/log_test_%s_%.2f.txt' % (taskname, ratio), 'a') as f:
                f.write('\t'.join(map(str, [cur_epoch, Precison, Recall, F1])) + '\n')

    # After training, test on test set

    print('final results')
    Precison, Recall, F1 = check_without_fine_tuning(batch_size, best_feature, best_crf, test_check_data, device)

    return Precison, Recall, F1


def update_params_from_args():
    parser = argparse.ArgumentParser(description='NER')
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--task', type=str, default=r'bionlp13pc')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # data_file = [('conll2003', 1000, 1000),
    #              ('ontonotes5', 1000, 1000),
    #              ('wnut17', 500, 500),
    #              ('wikigold', 1000, 1000),
    #              ('bionlp13pc', 500, 500),
    #              ('re3d', 50, 100),
    #              ('mitmovieeng', 500, 500),
    #              ('mitrestaurant', 500, 500),
    #              ('sec', 100, 300)

    testparams = update_params_from_args()
    data_ratio = [1, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    for cur_ratio in data_ratio:
        gpu_no = 0
        CUDA_VISIBLE_DEVICES = "%d" % gpu_no
        DEVICE = torch.device("cuda:%d" % gpu_no)

        model_path = r'/MetaNER/output/heter/our_heter_1order_4_sources_2019_09_28_03_13_13'
        target_name = testparams.task
        model_input_path = r'/MetaNER/input/heter_exp/' + target_name  #
        FINE_TUNING = True
        LOAD_CRF = False
        CombineFlag = False
        b_size = 16
        data_per = cur_ratio

        result_csv_path = r'/MetaNER/output/heter'

        test_check_data = get_target_input_data_test(model_input_path)

        if FINE_TUNING == True:
            train_check_data = get_target_input_data_train(model_input_path, data_per)
            dev_check_data = get_target_input_data_dev(model_input_path)

        params_path = os.path.join(model_path, 'a_params.pickle')
        params = pickle.load(open(params_path, 'rb'))

        if CombineFlag == True:
            TAG_TO_NUNBER, RE_PATTENS = get_TAG_and_pattens('single_combine')
            cur_total_entity_type = len(TAG_TO_NUNBER)
        else:
            TAG_TO_NUNBER, RE_PATTENS = get_TAG_and_pattens('single_' + target_name)
            cur_total_entity_type = len(TAG_TO_NUNBER)

        feature_extractor = SeqFeatureExtractor(params, DEVICE).to(DEVICE)
        crf_layer = MyCRF(params.rnn_direction_no * params.encoder_rnn_hidden, cur_total_entity_type, DEVICE).to(DEVICE)

        load_dic = torch.load(os.path.join(model_path, 'model_best_dev.torchsave'),
                              map_location=lambda storage, loc: storage.cuda(gpu_no))

        feature_extractor.load_state_dict(load_dic['f_model'])

        if LOAD_CRF == True:
            crf_layer.load_state_dict(load_dic['crf'])

        if FINE_TUNING == False:
            P, R, F1 = check_without_fine_tuning(b_size, feature_extractor, crf_layer, test_check_data, DEVICE)
        else:
            best_f = SeqFeatureExtractor(params, DEVICE).to(DEVICE)
            best_crf = MyCRF(params.rnn_direction_no * params.encoder_rnn_hidden, cur_total_entity_type, DEVICE).to(
                DEVICE)
            P, R, F1 = check_with_fine_tuning(target_name, data_per, b_size, best_f, best_crf, feature_extractor,
                                              crf_layer, train_check_data, dev_check_data, test_check_data, DEVICE)

        with open(os.path.join(result_csv_path, 'heter_our_approach_data_ratio_%s.txt' % target_name), 'a') as f:
            f.write('\t'.join(map(str, [target_name, data_per, P, R, F1])) + '\n')
