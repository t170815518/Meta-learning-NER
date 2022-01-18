

# for command line
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch

import random
import numpy as np
from torch.backends import cudnn

import argparse





def str2bool(v):
  #susendberg's function
  return v.lower() in ("yes", "true", "t", "1")



def update_params_from_args():
    parser = argparse.ArgumentParser(description='NER')
    parser.add_argument('--task', type=str, default=r'base')
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--variant', type=str, default=r'our')
    parser.add_argument('--foldpath', type=str, default=r'/MetaNER/input/heter_exp')
    parser.add_argument('--save_path', type=str,default=r'/MetaNER/output/heter_single')
    parser.add_argument('--no_train_domain', type=int, default=1)
    parser.add_argument('--single_domain', type=str, default='mitmovieeng') #bionlp13pc  re3d mitrestaurant sec


    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--cudnn', type=bool, default=True)
    parser.add_argument('--char_cnn_flag', type=str2bool, default=True)
    parser.add_argument('--encoder_rnn_layers', type=int, default=1)
    parser.add_argument('--encoder_rnn_hidden', type=int, default=128)
    parser.add_argument('--nn_dropout', type=float, default=0.2)
    parser.add_argument('--rnn_dropout', type=float, default=0.5)
    parser.add_argument('--fix_word_embedding', type=str2bool, default=False)
    parser.add_argument('--dim_of_word', type=int, default=300)
    parser.add_argument('--dim_of_char', type=int, default=100)
    parser.add_argument('--dim_of_char_embedding', type=int, default=100)
    parser.add_argument('--rnn_direction_no', type=int, default=2)
    parser.add_argument('--sample_train_for_each_domain', type=int, default=500)
    parser.add_argument('--alphabet_size', type=int, default=591)
    parser.add_argument('--word_size', type=int, default=70771)
    parser.add_argument('--no_epoch', type=int, default=70)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--word_embedding_lr', type=float, default=1e-4)
    parser.add_argument('--check_ecah_epoch', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=1e-4)
    parser.add_argument('--heldout_p', type=float, default=1.0)
    parser.add_argument('--theta_update', type=str, default=r'all')
    parser.add_argument('--lr_decay', type=bool, default=True)



    args = parser.parse_args()


    # for homo, the embeddings are same
    if args.single_domain == 'combine':
        args.init_embeddings_path = r'/MetaNER/input/heter_exp/combine/save_initializaton_embeddings.pickle'
    else:
        args.init_embeddings_path = r'/MetaNER/input/heter_exp/save_initializaton_embeddings.pickle'

    return args



if __name__ == '__main__':
    #CUDA_VISIBLE_DEVICES = "6"

    seed_num = 50
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    DEVICE = torch.device("cuda")
    params = update_params_from_args()

    cudnn.enabled = params.cudnn

    if params.variant =='our':
        from solver_meta_ner_heter_our import MetaNERHeter_baseline, MetaNERHeter
    if params.variant == 'maml':
        pass




    if params.task == 'base':

        solver_obj = MetaNERHeter_baseline(params,DEVICE)
        solver_obj.train()

    elif params.task == 'notbase':

        solver_obj = MetaNERHeter(params, DEVICE)
        solver_obj.train()