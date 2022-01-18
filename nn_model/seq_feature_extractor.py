
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from nn_model.wordrepresenter import WordRepresenter
import torch.nn.utils.rnn as R

class SeqFeatureExtractor(nn.Module):
    def __init__(self,params,DEVICE):
        super(SeqFeatureExtractor, self).__init__()
        print('SeqFeatureExtractor')

        self.params = params

        self.nnDropout = nn.Dropout(params.nn_dropout)
        self.num_encoder_rnn_layers = params.encoder_rnn_layers

        input_dim = params.dim_of_word + params.dim_of_char
        self.encoder_hidden_size = params.encoder_rnn_hidden
        self.encoder_rnn_layers = params.encoder_rnn_layers

        self.encoder_rnn = getattr(nn, 'GRU')(input_size=input_dim,
                                                       hidden_size=params.encoder_rnn_hidden,
                                                       num_layers=params.encoder_rnn_layers,
                                                       bidirectional=False if self.params.rnn_direction_no == 1 else True,
                                                       dropout=params.rnn_dropout,
                                                       batch_first=True)
        self.device = DEVICE

        self.wordrep = WordRepresenter(params, DEVICE)

        ignored_params = list(map(id, self.wordrep.word_embed.parameters()))
        self.base_params = filter(lambda p: id(p) not in ignored_params, self.parameters())

        self.type_loss_function = nn.NLLLoss()



    def initHidden(self, batchsize):  # no batch first setting

        h_0 = torch.zeros(self.params.rnn_direction_no * self.encoder_rnn_layers, batchsize,
                          self.encoder_hidden_size).to(self.device)
        # h_0 of shape(num_layers * num_directions, batch, hidden_size):
        return h_0





    def _run_rnn_packed(self, cell, x, x_lens, h=None):
        x_packed = R.pack_padded_sequence(x, x_lens, batch_first=True)

        if h is not None:
            output, h = cell(x_packed, h)
        else:
            output, h = cell(x_packed)

        output, _ = R.pad_packed_sequence(output, batch_first=True)

        return output, h

    def forward(self, inputTensors):
        char_input = inputTensors[2]
        word_input = inputTensors[1]
        lens = inputTensors[3]

        X = self.wordrep(word_input, char_input)

        batch_size = X.shape[0]

        # RNN
        h_pass = self.initHidden(batch_size)

        o, h = self._run_rnn_packed(self.encoder_rnn, X, lens, h_pass)  # batch_first=True
        o = o.contiguous()

        o = self.nnDropout(o)



        #o: output of shape (seq_len, batch, num_directions * hidden_size)
        #h: h_n of shape (num_layers * num_directions, batch, hidden_size)
        # Like output, the layers can be separated using h_n.view(num_layers, num_directions, batch, hidden_size).

        return o, h








