import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nn_model.charcnn import CharCNN
import pickle
import os



class WordRep(nn.Module):
    def __init__(self,params,DEVICE):
        super(WordRep, self).__init__()
        print('Word Embedding + Character CNN')

        self.char_cnn_flag = params.char_cnn_flag
        if self.char_cnn_flag == True:
            self.char_CNN = CharCNN(params,DEVICE)

        self.word_embed = nn.Embedding(params.word_size, params.dim_of_word)

        self.nnDropout = nn.Dropout(params.nn_dropout)

        load_embeddings = pickle.load(open(params.init_embeddings_path,'rb'))
        if params.fix_word_embedding == True:
            self.word_embed.weight.data.copy_(torch.from_numpy(load_embeddings).to(DEVICE))
            self.word_embed.weight.requires_grad = False

        else:
            self.word_embed.weight.data.copy_(torch.from_numpy(load_embeddings).to(DEVICE))
            self.word_embed.weight.requires_grad = True



    def forward(self, X_word, X_char):

        if self.char_cnn_flag == True:
            char_features = self.char_CNN(X_char) #(N,Lw,Lc)  Lw: no. of words. Lc: no. of characters

            X = self.word_embed(X_word) #X: [N,Lw]-->[N,Lw,D]

            X = torch.cat((X,char_features),2)
            X = self.nnDropout(X)
        else:
            X = self.word_embed(X_word)
            X = self.nnDropout(X)

        return X














