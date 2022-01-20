"""
CNN_BiGRU, low-coupling module implementation
"""

import torch
import logging
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.init import xavier_uniform_
from torchcrf import CRF


class CNN_BiGRU(nn.Module):
    """
    feature extractor, encoder
    ref:
        - https://learning.oreilly.com/library/view/the-deep-learning/9781838989217/
        - Li, J., et al. (2020). MetaNER: Named Entity Recognition with Meta-Learning. International World Wide Web
        Conference(WWW). Taipei, Taiwan. (Special thanks to Dr. Li Jing for providing source codes)
    """

    def __init__(self, word_size: int, word_emb_dim, alphabet_size, char_emb_dim, hidden_size, word_pad_idx,
                 char_pad_idx, is_freeze: bool, cnn_total_num: int, dropout: float, pretrained_path: str = None,
                 layer_num: int = 1):
        """
        :param word_size: int, the size of unique word in the corpus (e.g. GloVe corpus)
        :param word_emb_dim: int, the size of embedding for word vector
        :param char_emb_dim: int, the size of embedding for character vector
        :param hidden_size: int, the size of UNI-DIRECTION hidden state dimension as the output of RNN
        :param layer_num: int, the number of layer of RNN layer
        :param word_pad_idx: non-negative int, the index for padding word <PAD>
        :param alphabet_size: positive int, the size of unique characters
        :param char_pad_idx: non-negative int, the index for padding character
        :param is_freeze: bool, whether to fine-tune the pre-trained word vector
        :param cnn_total_num: int, the total number of CNN filters in all sizes
        :param dropout: float, dropout rate to avoid over-fitting
        :param pretrained_path: str, path to a numpy array
        todo: pre-process the word2vec
        """
        super(CNN_BiGRU, self).__init__()

        # 1. for character-level feature
        self.char_embedding = nn.Embedding(alphabet_size + 1, char_emb_dim, padding_idx=char_pad_idx)
        xavier_uniform_(self.char_embedding.weight)
        #   various sizes of CNN filters
        cnn_num = int(cnn_total_num / 4)
        self.conv1 = nn.Conv2d(1, cnn_num, (2, char_emb_dim))
        self.conv2 = nn.Conv2d(1, cnn_num, (3, char_emb_dim))
        self.conv3 = nn.Conv2d(1, cnn_num, (4, char_emb_dim))
        self.conv4 = nn.Conv2d(1, cnn_num, (5, char_emb_dim))

        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)

        # 2. for word-level feature
        if pretrained_path is None:  # initialize the embedding
            self.word_embedding = nn.Embedding(word_size, word_emb_dim, padding_idx=word_pad_idx)
            xavier_uniform_(self.word_embedding.weight)
        else:
            emb_arr = torch.load(pretrained_path)
            appended_emb_arr = torch.zeros([emb_arr.shape[0] + 1, emb_arr.shape[1]])
            appended_emb_arr[:-1, :] = emb_arr
            self.word_embedding = nn.Embedding.from_pretrained(appended_emb_arr, is_freeze, padding_idx=word_pad_idx)

        # 3. BiGRU
        self.rnn_layer = nn.GRU(input_size=char_emb_dim + word_emb_dim, hidden_size=hidden_size, num_layers=layer_num,
                                batch_first=True, dropout=0.5, bidirectional=True)

    def forward(self, words, characters) -> torch.tensor:
        """
        For each sequence, the hidden state is initialized to none.
        This is because, in each sequence, the network will try to map the inputs to the targets (when given a set of
        parameters). This mapping should occur without any bias (hidden state) from the previous runs through the
        dataset.
        :param words: LongTensor, word_id matching the word_embedding, shape = [batch, seq_len]
        :param characters: LongTensor, the ids of characters, shape = [batch, seq_len, padded_char_seq]
        :return: torch.tensor, the representation of the batch
        """
        # get word feature
        word_features = self.word_embedding(words)

        # get char feature
        char_embedding = self.__get_char_embedding(characters)

        in_feature = torch.cat([word_features, char_embedding], dim=-1)
        encoded_features, h_final = self.rnn_layer(in_feature, None)

        return encoded_features, h_final

    def __get_char_embedding(self, characters):
        """
        Pass the ids of characters through embedding and a series of CNN filters to get the final character embedding
        :param characters: LongTensor, the ids of characters, shape = [batch, seq_len, padded_char_seq]
        :return: torch.tensor, the character-level feature
        """
        char_emb = self.char_embedding(characters)
        char_emb = self.dropout(char_emb)

        batch_size = char_emb.shape[0]
        seq_len = char_emb.shape[1]
        padded_char_seq_len = char_emb.shape[2]
        char_emb_dim = char_emb.shape[3]

        # pass through CNN filters with various sizes
        x = char_emb.reshape(batch_size * seq_len, 1, padded_char_seq_len, char_emb_dim)  # (N,Cin,Hin,Win)
        x1 = functional.relu(self.conv1(x))  # (N*seq_len, Filters, H_out,1)
        x1 = self.pool(x1)  # (N*seq_len, Filters,1,1)
        x1 = x1.reshape(batch_size, seq_len, -1)  # (N,seq_len,25)
        x2 = functional.relu(self.conv2(x))
        x2 = self.pool(x2)
        x2 = x2.reshape(batch_size, seq_len, -1)
        x3 = functional.relu(self.conv3(x))
        x3 = self.pool(x3)
        x3 = x3.reshape(batch_size, seq_len, -1)
        x4 = functional.relu(self.conv4(x))
        x4 = self.pool(x4)
        x4 = x4.reshape(batch_size, seq_len, -1)
        Y = torch.cat((x1, x2, x3, x4), 2)
        return Y


def reverse_gradient(grad):
    new_grad = grad * -1
    logging.info("grad is reversed.")
    return new_grad


class MLP_DomainDiscriminator(nn.Module):
    """
    MLP-based network as the domain discriminator
    """

    def __init__(self, feature_dim: int, domain_num: int, device):
        super(MLP_DomainDiscriminator, self).__init__()
        self.attention = torch.zeros((feature_dim, 1), requires_grad=True, device=device)
        self.linear = nn.Linear(feature_dim, domain_num).to(device)

        self.constant = -1  # the constant multiplied to gradient during gradient reverse operation
        # initialize weights
        xavier_uniform_(self.attention)

        # loss
        self.loss_func = nn.NLLLoss()

    def forward(self, feature):
        x = feature * 1
        # backward hook, called during back-propagation to reverse gradient
        x.register_hook(reverse_gradient)

        attention_score = torch.matmul(x, self.attention).squeeze()
        attention_score = functional.softmax(attention_score).view(x.size(0), x.size(1), 1)
        scored_x = x * attention_score  # [20,40,256]  [20,40,1]   --> [20,40,256]
        condensed_x = torch.sum(scored_x, dim=1)  # [20,256]
        domain_output = functional.log_softmax(self.linear(condensed_x), 1)
        return domain_output

    def loss(self, feature, true_tag):
        x = self.forward(feature)
        l = self.loss_func(x, true_tag)
        return l


class Decoder(nn.Module):
    def __init__(self, feature_dim: int, tag_num, is_use_crf: bool = True):
        super(Decoder, self).__init__()

        self.is_use_crf = is_use_crf
        self.tag_num = tag_num
        self.linear = nn.Linear(feature_dim, tag_num)
        if is_use_crf:
            self.crf = CRF(tag_num, batch_first=True)

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def loss(self, encoded_features, true_tags):
        """

        :param encoded_features:
        :param true_tags:
        :return: the returned value is the log likelihood so you’ll need to make this value negative
        as the loss to optimize
        """
        x = self.linear(encoded_features)  # [BATCH, SEQ_LEN, TAG_NUM]
        real_entries_mask = true_tags != -1  # to mask out the padding entries
        if self.is_use_crf:
            l = -self.crf(x, true_tags, mask=real_entries_mask, reduction='mean')
        else:
            x = x.view(-1, self.tag_num)
            real_entries_mask = real_entries_mask.view(-1)
            true_tags = true_tags.view(-1)
            x = x[real_entries_mask]
            true_tags = true_tags[real_entries_mask]
            l = self.cross_entropy_loss(x, true_tags)
        return l

    def decode(self, encoded_features, mask):
        """

        :param encoded_features:
        :param mask:
        :return: list of list
        """
        x = self.linear(encoded_features)
        if self.is_use_crf:
            tag = self.crf.decode(x, mask=mask)
        else:
            tag = []
            pred_tag = torch.argmax(x, dim=-1)
            batch_size, seq_max_len = pred_tag.size()
            for i in range(batch_size):
                t = []
                for j in range(seq_max_len):
                    if mask[i, j]:
                        t.append(int(pred_tag[i, j]))
                    else:
                        break
                tag.append(t)
        return tag
