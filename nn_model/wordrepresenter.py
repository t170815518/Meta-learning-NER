import pickle

import torch
import torch.nn as nn
import numpy as np

from nn_model.charcnn import CharCNN


class WordRepresenter(nn.Module):
    """
    Represents the word as vectors via pre-trained word embedding and character-level features
    """
    def __init__(self, params, DEVICE):
        super(WordRepresenter, self).__init__()
        self.char_cnn_flag = params.char_cnn_flag
        if self.char_cnn_flag:
            self.char_CNN = CharCNN(params, DEVICE)
        self.word_embed = nn.Embedding(params.word_size, params.dim_of_word)
        self.nnDropout = nn.Dropout(params.nn_dropout)
        load_embeddings = np.load(params.init_embeddings_path).astype(float)
        load_embeddings = torch.tensor(load_embeddings)

        self.word_embed = nn.Embedding.from_pretrained(load_embeddings, freeze=params.fix_word_embedding)

    def forward(self, X_word, X_char):
        if self.char_cnn_flag:
            char_features = self.char_CNN(X_char)  # (N,Lw,Lc)  Lw: no. of words. Lc: no. of characters
            X = self.word_embed(X_word)  # X: [N,Lw]-->[N,Lw,D]
            X = torch.cat((X, char_features), 2)
            X = self.nnDropout(X)
        else:
            X = self.word_embed(X_word)
            X = self.nnDropout(X)
        return X
