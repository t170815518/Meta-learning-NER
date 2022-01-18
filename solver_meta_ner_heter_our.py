
from utils.functions import sample_sorted_alignment_minibatch_from_numpy, select_data_based_on_select_index
from utils.functions import sample_sorted_alignment_minibatch_from_numpy_with_domain_y, fix_nn

import pickle
import os
import random
import numpy as np

from nn_model.seq_feature_extractor import SeqFeatureExtractor
from nn_model.crflayer import MyCRF
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import time
from utils.BIOES_ENCODER import get_TAG_and_pattens
from utils.nermetric import get_metric_number_with_RE, get_ner_seg_score_from_listmatrix
import torch
from nn_model.discriminator import DomainDiscriminator
import torch.nn as nn
from collections import OrderedDict

from torch.optim.lr_scheduler import StepLR

class MetaNERHeter_baseline(object):
    def __init__(self, params,DEVICE):
        self.batch_size = params.batch_size
        self.device = DEVICE
        self.params =  params

        self.init_network()



    def init_network(self):


        if self.params.no_train_domain > 1:
            self.train_domains = ['conll2003','ontonotes5','wikigold','wnut17']
        else:
            self.train_domains = [self.params.single_domain]


        self.feature_extractor = SeqFeatureExtractor(self.params, self.device).to(self.device)

        #load numpy data
        self.train_data = {}
        self.dev_data = {}
        self.evaluate_train_data = {}

        self.RE_PATTENS = {}
        self.TAG_TO_NUNBER = {}
        self.all_crf = {}
        for name in self.train_domains:


            TAG_TO_NUNBER, RE_PATTENS = get_TAG_and_pattens('single_'+name)
            self.RE_PATTENS[name] = RE_PATTENS
            self.TAG_TO_NUNBER[name] = TAG_TO_NUNBER
            cur_total_entity_type = len(TAG_TO_NUNBER)
            self.all_crf[name] = MyCRF(self.params.rnn_direction_no * self.params.encoder_rnn_hidden, cur_total_entity_type, self.device).to(self.device)




            train_x_word = pickle.load(open(os.path.join(self.params.foldpath+'/%s'%name, 'train_x_word_mapping'), 'rb'))
            train_x_char = pickle.load(open(os.path.join(self.params.foldpath+'/%s'%name, 'train_x_char_mapping'), 'rb'))
            train_y = pickle.load(open(os.path.join(self.params.foldpath+'/%s'%name, 'train_y.pickle'), 'rb'))

            self.train_data[name] = [train_x_word, train_x_char, train_y]

            dev_x_word = pickle.load(open(os.path.join(self.params.foldpath+'/%s'%name, 'dev_x_word_mapping'), 'rb'))
            dev_x_char = pickle.load(open(os.path.join(self.params.foldpath+'/%s'%name, 'dev_x_char_mapping'), 'rb'))
            dev_y = pickle.load(open(os.path.join(self.params.foldpath+'/%s'%name, 'dev_y.pickle'), 'rb'))
            self.dev_data[name] = [dev_x_word, dev_x_char, dev_y]


        self.sample_train_data()




    def sample_train_data(self):

        self.evaluate_train_data= {}


        for cur_domain in self.train_domains:
            train_x_word, train_x_char, train_y = self.train_data[cur_domain]
            ori_tags, sentence_length_Y, label_Y = train_y
            select_index = np.array(random.sample(range(len(train_x_word)), self.params.sample_train_for_each_domain))

            re_x_word = train_x_word[select_index]
            re_x_char = train_x_char[select_index]

            ori_tags = np.array(ori_tags)  # ori_tags is a list

            re_y_1,  re_y_2, re_y_3= ori_tags[select_index], sentence_length_Y[select_index], label_Y[select_index]

            dev_y = re_y_1,  re_y_2, re_y_3

            self.evaluate_train_data[cur_domain] = [re_x_word, re_x_char, dev_y]




    def save_params(self,params,savepath):

        ALL_Attributes = list(filter(lambda a: not a.startswith('_'), dir(params)))


        with open(os.path.join(savepath, 'a_params_tracking.txt'), 'a') as f:
            f.write('training size: %d, dev size: %d '%(sum(len(self.train_data[cur_name][0]) for cur_name in self.train_domains),sum(len(self.dev_data[cur_name][0]) for cur_name in self.train_domains)) + '\n')

        with open(os.path.join(savepath, 'a_params_tracking.txt'), 'a') as f:
            for cur_a in ALL_Attributes:
                f.write(':'.join(map(str, (cur_a, getattr(params,cur_a)))) +'\n' )

        with open(os.path.join(savepath, 'a_params.pickle'), 'wb') as f:
            pickle.dump(self.params, f)




    def check_performance(self,check_data):


        self.feature_extractor.eval()


        all_metric_matrix = []

        return_list = []

        for cur_domain in self.train_domains:


            self.all_crf[cur_domain].eval()

            cur_check_data = check_data[cur_domain]

            print('check domain performance')
            #=============split data ===============
            data_L = len(cur_check_data[0])

            need_loop = int(np.ceil(data_L / self.batch_size))

            metric_matrix = []

            for lp in range(need_loop):
                print('check %d of %d'%(lp,need_loop))
                startN = lp*self.batch_size
                endN =  (lp+1)*self.batch_size
                if endN > data_L:
                    endN = data_L



                select_index  = np.array(list(range(startN,endN)))

                select_data = select_data_based_on_select_index(cur_check_data,select_index)

                inputTensors = sample_sorted_alignment_minibatch_from_numpy(select_data, self.batch_size,sampleFlag=False,device= self.device)


                encoder_features, combined_h_features = self.feature_extractor(inputTensors)
                scores, paths = self.all_crf[cur_domain].viterbi_decode(encoder_features, inputTensors)

                g_seq = inputTensors[4]
                lens = inputTensors[3]
                batch_metric_number = get_metric_number_with_RE(g_seq, paths, lens, self.RE_PATTENS[cur_domain])

                metric_matrix.extend(batch_metric_number)
                all_metric_matrix.extend(batch_metric_number)

            print('check current domain performance size:', len(metric_matrix))
            domainP, domainR, domainF = get_ner_seg_score_from_listmatrix(metric_matrix)

            return_list.extend([domainP, domainR, domainF])


        print('check all domain performance size:',len(all_metric_matrix))
        all_P, all_R, all_F = get_ner_seg_score_from_listmatrix(all_metric_matrix)
        return_list.extend([all_P, all_R, all_F])

        #=============check out ======================
        return return_list


    def train(self):

        date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        cur_save_path = os.path.join(self.params.save_path, 'heter_base_%d_sources_'%self.params.no_train_domain + date_time)
        os.makedirs(cur_save_path)
        self.save_params(self.params, cur_save_path)



        self.opt_all_exc_crf = optim.Adam([{'params': self.feature_extractor.wordrep.word_embed.parameters(), 'lr': self.params.word_embedding_lr, 'weight_decay':0},
                                   {'params':self.feature_extractor.base_params}],
                                       lr=self.params.lr, weight_decay=self.params.weight_decay)

        self.opt_only_crf = {}
        for name in self.train_domains:
            self.opt_only_crf[name] = optim.Adam([{'params': self.all_crf[name].parameters()}],
                                          lr=self.params.lr, weight_decay=self.params.weight_decay)




        num_each_epoch = int(np.round(sum(len(self.train_data[cur_name][0]) for cur_name in self.train_domains) / (self.batch_size*len(self.train_domains))))
        total_iter = num_each_epoch * self.params.no_epoch

        check_point_iter_no = int(num_each_epoch / self.params.check_ecah_epoch) - 2

        best_train_F1 = 0
        best_dev_F1 = 0

        for iter in range(total_iter):

            cur_epoch = int(np.ceil(iter / num_each_epoch + 0.00000001))
            cur_iter = iter % num_each_epoch
            print("epoch:%d/%d,iteration:%d/%d" % (cur_epoch, self.params.no_epoch, cur_iter, num_each_epoch))


            for cur_domain in self.train_domains:

                inputTensors = sample_sorted_alignment_minibatch_from_numpy(self.train_data[cur_domain], self.batch_size,
                                                                        sampleFlag=True, device=self.device)

                self.feature_extractor.train()
                self.all_crf[cur_domain].train()


                encoder_features, combined_h_features = self.feature_extractor(inputTensors)
                crf_loss = self.all_crf[cur_domain].crf_neglog_loss(encoder_features,inputTensors)

                self.opt_all_exc_crf.zero_grad()
                self.opt_only_crf[cur_domain].zero_grad()
                crf_loss.backward()

                clip_grad_norm_(list(self.feature_extractor.wordrep.word_embed.parameters())+
                                list(self.feature_extractor.base_params)+
                                list(self.all_crf[cur_domain].parameters()), 5)
                self.opt_all_exc_crf.step()
                self.opt_only_crf[cur_domain].step()

                trackloss = [cur_epoch, iter, crf_loss.item()]
                with open(os.path.join(cur_save_path, 'a_loss_tracking.txt'), 'a') as f:
                    f.write('\t'.join(map(str, trackloss)) + '\n')



            #check performance
            if iter % check_point_iter_no == 0 and iter != 0:

                SeqPRF = self.check_performance(self.evaluate_train_data)
                trackperformace = ['train', cur_epoch, iter] + SeqPRF
                with open(os.path.join(cur_save_path, 'a_performance_tracking.txt'), 'a') as f:
                    f.write('\t'.join(map(str, trackperformace)) + '\n')


                if best_train_F1 < SeqPRF[-1]:
                    best_train_F1 = SeqPRF[-1]




                SeqPRF = self.check_performance(self.dev_data)
                trackperformace = ['dev',cur_epoch,iter] + SeqPRF
                with open(os.path.join(cur_save_path, 'a_performance_tracking.txt'), 'a') as f:
                    f.write('\t'.join(map(str, trackperformace)) + '\n')


                if best_dev_F1 < SeqPRF[-1]:
                    best_dev_F1 = SeqPRF[-1]


                    if len(self.train_domains)==1:
                        torch.save({'f_model': self.feature_extractor.state_dict(),
                                    'crf': self.all_crf[self.params.single_domain].state_dict()},
                                   os.path.join(cur_save_path, r'model_best_dev.torchsave'))
                    else:
                        torch.save({'f_model': self.feature_extractor.state_dict()},
                                   os.path.join(cur_save_path, r'model_best_dev.torchsave'))





class MetaNERHeter(MetaNERHeter_baseline):
    def __init__(self, params,DEVICE):
        super(MetaNERHeter, self).__init__(params,DEVICE)


        print('init new adversarial layer')
        self.domain_classifier = DomainDiscriminator(params).to(DEVICE)

        self.train_domains_index = {}
        for i, name in enumerate(self.train_domains):
            self.train_domains_index[name] = i

        self.adv_loss_function = nn.NLLLoss()

        self.device = DEVICE
        self.params = params


        self.temp_new_feature_extractor = SeqFeatureExtractor(self.params, self.device).to(self.device)
        self.temp_old_feature_extractor = SeqFeatureExtractor(self.params, self.device).to(self.device)




    def train(self):

        date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        cur_save_path = os.path.join(self.params.save_path, 'our_heter_1order_4_sources_' + date_time)
        os.makedirs(cur_save_path)
        self.save_params(self.params, cur_save_path)




        self.opt_feature = optim.Adam([{'params': self.feature_extractor.wordrep.word_embed.parameters(), 'lr': self.params.word_embedding_lr, 'weight_decay':0},
                                   {'params':self.feature_extractor.base_params}],
                                       lr=self.params.lr, weight_decay=self.params.weight_decay)



        self.opt_only_crf = {}
        for name in self.train_domains:
            self.opt_only_crf[name] = optim.Adam([{'params': self.all_crf[name].parameters()}],
                                                 lr=self.params.lr, weight_decay=self.params.weight_decay)


        self.opt_adv = optim.Adam([{'params':self.domain_classifier.parameters()}],
                                       lr=self.params.lr, weight_decay=self.params.weight_decay)


        if self.params.lr_decay == True:
            scheduler_f = StepLR(self.opt_feature, step_size=10, gamma=0.98)
            scheduler_adv = StepLR(self.opt_adv, step_size=10, gamma=0.98)


            scheduler_crf = {}
            for name in self.train_domains:
                scheduler_crf[name] = StepLR(self.opt_only_crf[name], step_size=10, gamma=0.98)



        num_each_epoch = int(np.round(sum(len(self.train_data[cur_name][0]) for cur_name in self.train_domains) / (
                    self.batch_size * len(self.train_domains))))
        total_iter = num_each_epoch * self.params.no_epoch

        check_point_iter_no = int(num_each_epoch / self.params.check_ecah_epoch) - 2

        best_train_F1 = 0
        best_dev_F1 = 0

        pre_epoch = 1

        for iter in range(total_iter):

            cur_epoch = int(np.ceil(iter / num_each_epoch + 0.00000001))
            cur_iter = iter % num_each_epoch
            print("epoch:%d/%d,iteration:%d/%d" % (cur_epoch, self.params.no_epoch, cur_iter, num_each_epoch))



            tempdos = np.random.permutation(self.train_domains)
            meta_train_domains = tempdos[0:-1]
            meta_val_domains = tempdos[-1:]

            meta_train_loss_main = 0.0
            meta_train_loss_adv = 0.0
            meta_loss_val = 0.0

            self.opt_feature.zero_grad()

            for name in self.train_domains:
                self.opt_only_crf[name].zero_grad()

            self.opt_adv.zero_grad()
            #self.domain_classifier.grad.zero_()

            #Loop meta-train
            for cur_domain in meta_train_domains:
                inputTensors = sample_sorted_alignment_minibatch_from_numpy_with_domain_y(self.train_domains_index[cur_domain],self.train_data[cur_domain],
                                                                            self.batch_size,
                                                                            sampleFlag=True, device=self.device)

                self.feature_extractor.train()
                self.all_crf[cur_domain].train()

                # the crf loss
                encoder_features, combined_h_features = self.feature_extractor(inputTensors)
                crf_loss = self.all_crf[cur_domain].crf_neglog_loss(encoder_features, inputTensors)
                meta_train_loss_main += crf_loss

                # the adv loss
                self.domain_classifier.train()
                domain_output = self.domain_classifier.get_discriminator_log_output(encoder_features, 1.0)
                adv_loss = self.adv_loss_function(domain_output, inputTensors[-1])
                meta_train_loss_adv += adv_loss




            temp_f_weights = OrderedDict((name, param) for (name, param) in self.feature_extractor.named_parameters())

            temp_crf_weights = {}
            for name in self.train_domains:
                temp_crf_weights[name] = OrderedDict((name, param) for (name, param) in self.all_crf[name].named_parameters())


            temp_adv_weights = OrderedDict((name, param) for (name, param) in self.domain_classifier.named_parameters())



            # temp_f_weights_old make correct prediction
            grads_f_old = torch.autograd.grad(meta_train_loss_main, temp_f_weights.values(), retain_graph=True)  #we do not take second order , create_graph=True
            temp_f_weights_old =  OrderedDict((name, param - self.params.alpha * grad) for ((name, param), grad) in zip(temp_f_weights.items(), grads_f_old))




            #temp_f_weights_new makes error prediction
            grads_f_new = torch.autograd.grad(meta_train_loss_adv, temp_f_weights.values(), retain_graph=True)  #, create_graph=True

            if self.params.theta_update =='all':
                temp_f_weights_new = OrderedDict((name, param - self.params.alpha * grad) for ((name, param), grad) in zip(temp_f_weights_old.items(), grads_f_new))
            if self.params.theta_update =='ori':
                temp_f_weights_new = OrderedDict((name, param - self.params.alpha * grad) for ((name, param), grad) in zip(temp_f_weights.items(), grads_f_new))




            self.temp_new_feature_extractor.load_state_dict(temp_f_weights_new)
            self.temp_new_feature_extractor.train()


            self.temp_old_feature_extractor.load_state_dict(temp_f_weights_old)
            self.temp_old_feature_extractor.train()
         




            for cur_val_domain in meta_val_domains:
                inputTensors = sample_sorted_alignment_minibatch_from_numpy_with_domain_y(
                    self.train_domains_index[cur_val_domain], self.train_data[cur_val_domain],
                    self.batch_size,
                    sampleFlag=True, device=self.device)

                old_encoder_features, old_combined_h_features = self.temp_old_feature_extractor(inputTensors)
                new_encoder_features, new_combined_h_features = self.temp_new_feature_extractor(inputTensors)

                crf_loss_old = self.all_crf[cur_val_domain].crf_neglog_loss(old_encoder_features, inputTensors)

                temp_o = self.domain_classifier.get_discriminator_log_output(new_encoder_features, 1.0)
                adv_loss_meta = self.adv_loss_function(temp_o, inputTensors[-1])

                loss_held_out = crf_loss_old + adv_loss_meta



                meta_loss_val += loss_held_out * self.params.heldout_p





            #Calculate gradients
            grads_f_final = grads_f_old + grads_f_new
            grads_adv_final = torch.autograd.grad(meta_loss_val, temp_adv_weights.values(), retain_graph=True )

            grads_crf_final = {}
            for name in meta_train_domains:
                grads_crf_final[name] = torch.autograd.grad(meta_train_loss_main, temp_crf_weights[name].values(), retain_graph=True)

            for name in meta_val_domains:
                grads_crf_final[name] = torch.autograd.grad(meta_loss_val, temp_crf_weights[name].values(), retain_graph=True)




            #Gradient replacement

            for w_old, w_new in zip(self.feature_extractor.parameters(), grads_f_final):
                w_old.grad = w_new

            for name in self.train_domains:
                for w_old, w_new in zip(self.all_crf[name].parameters(), grads_crf_final[name]):
                    w_old.grad = w_new


            for w_old, w_new in zip(self.domain_classifier.parameters(), grads_adv_final):
                w_old.grad = w_new

            temp = []
            for name in self.train_domains:
                temp = temp + list(self.all_crf[name].parameters())
            clip_grad_norm_(list(self.feature_extractor.parameters())+temp+list(self.domain_classifier.parameters()), 5.0)


            self.opt_feature.step()
            for name in self.train_domains:
                self.opt_only_crf[name].step()
            self.opt_adv.step()


            # check performance
            if iter % check_point_iter_no == 0 and iter != 0:

                SeqPRF = self.check_performance(self.evaluate_train_data)
                trackperformace = ['train', cur_epoch, iter] + SeqPRF
                with open(os.path.join(cur_save_path, 'a_performance_tracking.txt'), 'a') as f:
                    f.write('\t'.join(map(str, trackperformace)) + '\n')

                if best_train_F1 < SeqPRF[-1]:
                    best_train_F1 = SeqPRF[-1]

                SeqPRF = self.check_performance(self.dev_data)
                trackperformace = ['dev', cur_epoch, iter] + SeqPRF
                with open(os.path.join(cur_save_path, 'a_performance_tracking.txt'), 'a') as f:
                    f.write('\t'.join(map(str, trackperformace)) + '\n')

                if best_dev_F1 < SeqPRF[-1]:
                    best_dev_F1 = SeqPRF[-1]

                    torch.save({'f_model': self.feature_extractor.state_dict()},
                               os.path.join(cur_save_path, r'model_best_dev.torchsave'))




            if (self.params.lr_decay == True) and (cur_epoch != pre_epoch):
                scheduler_f.step()
                for name in self.train_domains:
                    scheduler_crf[name].step()
                scheduler_adv.step()
                pre_epoch = cur_epoch














