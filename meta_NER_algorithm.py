import random
from collections import OrderedDict

import torch
from sklearn.metrics import f1_score
from torch import optim

from model import CNN_BiGRU, MLP_DomainDiscriminator, DomainCRF


MODEL_SAVE_PATH = "optimial_encoder.pt"


class Meta_NER_Trainer:
    """
    The wrapper class of solving Domain-adaptation NER with Meta-NER.
    Note there is some bug that only supports 1 valid domain
    To use the class, simply call:
        ```python
        trainer = Meta_NER_Trainer(...)
        trainer.train()
        ```
    """

    def __init__(self, batch_size: int, epoch_num: int, train_domains, inner_lr: float, lr: float, word_emb_lr: float,
                 weight_decay: float, eval_interval: int, alphabet_size: int, word_size, total_train_size: int,
                 char_emb_dim: int = 100, hidden_size: int = 128, word_emb_dim: int = 300):
        assert batch_size > 0
        assert epoch_num > 0
        assert inner_lr > 0.0
        assert lr > 0.0
        assert eval_interval > 0
        assert word_emb_lr > 0
        assert alphabet_size > 0
        assert total_train_size > 0

        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.train_domains = train_domains
        self.inner_lr = inner_lr
        self.eval_interval = eval_interval
        self.word_size = word_size
        self.word_emb_dim = word_emb_dim
        self.alphabet_size = alphabet_size
        self.char_emb_dim = char_emb_dim
        self.hidden_size = hidden_size
        self.total_sample_size = total_train_size

        # initialize the network components
        self.encoder = CNN_BiGRU(word_size, word_emb_dim, alphabet_size, char_emb_dim, hidden_size,
                                 word_pad_idx=word_size, char_pad_idx=alphabet_size, is_freeze=False, cnn_total_num=100,
                                 dropout=0.2, pretrained_path="pre_trained.pt")
        self.domain_discriminator = MLP_DomainDiscriminator(hidden_size * 2, len(train_domains))
        self.decoders = {}
        for domain in self.train_domains:
            self.decoders[domain.name] = DomainCRF(hidden_size * 2, domain.tag_num)
        # initialize the optimizer fixme: check here
        # todo: add weight decay
        self.encoder_optimizer = optim.Adam([{'params': self.encoder.word_embedding.parameters(),
                                              'lr': word_emb_lr, 'weight_decay': 0},
                                             {'params': self.encoder.conv1.parameters()},
                                             {'params': self.encoder.conv2.parameters()},
                                             {'params': self.encoder.conv3.parameters()},
                                             {'params': self.encoder.conv4.parameters()},
                                             {'params': self.encoder.rnn_layer.parameters()},
                                             {'params': self.encoder.char_embedding.parameters()}
                                             ],
                                            lr=lr, weight_decay=0)
        self.decoder_optimizers = {}
        for domain in self.train_domains:
            self.decoder_optimizers[domain.name] = optim.Adam([{'params': self.decoders[domain.name].parameters()}],
                                                              lr=lr, weight_decay=0)
        self.discriminator_optimizer = optim.Adam([{'params': self.domain_discriminator.parameters()}],
                                                  lr=lr, weight_decay=0)

    def train(self):
        """
        The core method of the class
        """
        # to approximate the iterations and epoch index
        iteration_per_epoch = int(self.total_sample_size / (self.batch_size * len(self.train_domains)))
        total_iteration_num = iteration_per_epoch * self.epoch_num

        best_train_f1 = 0
        best_dev_f1 = 0

        for i in range(total_iteration_num):
            # calculate the epoch and iteration id
            current_epoch = int(i / iteration_per_epoch)
            current_iteration = int(i % iteration_per_epoch)
            print("epoch:%d/%d,iteration:%d/%d" %
                  (current_epoch, self.epoch_num, current_iteration, iteration_per_epoch))

            # get meta-train and meta-val sets from train datasets, to simulate the domain shift from meta-train to
            # meta-val
            random.shuffle(self.train_domains)
            meta_train_domains = self.train_domains[:-1]
            meta_val_domain = self.train_domains[-1]

            meta_pred_loss, meta_adversial_loss = self.meta_train(meta_train_domains)

            # get the initial parameters of the models
            encoder_weights = OrderedDict((name, param) for (name, param) in self.encoder.named_parameters())
            decoder_weights = {}
            for domain in self.train_domains:
                decoder_weights[domain.name] = OrderedDict(
                        (name, param) for (name, param) in self.decoders[domain.name].named_parameters())

            domain_discriminator_weights = OrderedDict((name, param) for (name, param)
                                                       in self.domain_discriminator.named_parameters())
            # take the derivative
            grad_encoder_old = torch.autograd.grad(-meta_pred_loss, encoder_weights.values(), retain_graph=True)
            grad_encoder_new = torch.autograd.grad(meta_adversial_loss, encoder_weights.values(), retain_graph=True)
            # update the model with different derivatives
            encoder_old_params = OrderedDict((name, param - self.inner_lr * grad) for ((name, param), grad) in
                                             zip(encoder_weights.items(), grad_encoder_old))
            encoder_new_params = OrderedDict((name, param - self.inner_lr * grad) for ((name, param), grad) in
                                             zip(encoder_old_params.items(), grad_encoder_new))
            meta_valid_loss = self.meta_validate(encoder_old_params, encoder_new_params, meta_val_domain)

            # meta-optimization
            grads_encoder_final = grad_encoder_old + grad_encoder_new
            grads_discriminator_final = torch.autograd.grad(meta_valid_loss, domain_discriminator_weights.values(),
                                                            retain_graph=True)
            grads_decode_final = {}
            for domain in meta_train_domains:  # also update the encoder for meta-train domains
                grads_decode_final[domain.name] = torch.autograd.grad(meta_pred_loss + meta_adversial_loss,
                                                                      decoder_weights[domain.name].values(),
                                                                      retain_graph=True)
            if not isinstance(meta_val_domain, list):
                meta_val_domain = [meta_val_domain]
            for domain in meta_val_domain:
                grads_decode_final[domain.name] = torch.autograd.grad(meta_valid_loss,
                                                                      decoder_weights[domain.name].values(),
                                                                      retain_graph=True)
            for initial_param, final_grad in zip(self.encoder.parameters(), grads_encoder_final):
                initial_param.grad = final_grad

            for domain in self.train_domains:
                for initial_param, final_grad in zip(self.decoders[domain.name].parameters(),
                                                     grads_decode_final[domain.name]):
                    initial_param.grad = final_grad

            for initial_param, final_grad in zip(self.domain_discriminator.parameters(), grads_discriminator_final):
                initial_param.grad = final_grad

            # clip_norm
            self.encoder_optimizer.step()
            for domain in self.train_domains:
                self.decoder_optimizers[domain.name].step()
            self.discriminator_optimizer.step()

            # check-point
            if i % self.eval_interval == 0:
                pred_loss, f1 = self.evaluate(meta_train_domains)
                print("[meta_train] pred_loss={};f1={}".format(pred_loss, f1))
                if f1 > best_train_f1:
                    best_train_f1 = f1

                performance_str, f1 = self.evaluate(meta_val_domain)
                print("[meta_test] pred_loss={};f1={}".format(pred_loss, f1))
                if f1 > best_dev_f1:
                    best_dev_f1 = f1
                    torch.save(self.encoder.state_dict(), MODEL_SAVE_PATH)
                    print("best f1 found (), model is saved to {}".format(best_dev_f1, MODEL_SAVE_PATH))

    def meta_train(self, meta_train_domains):
        meta_train_pred_loss = 0.0
        meta_train_loss_adv = 0.0

        for current_domain in meta_train_domains:
            selected_samples = random.sample(current_domain.train, self.batch_size)
            (word_seq, char_seq), true_tags, domain_tag = current_domain.process_data(selected_samples)

            # set the model's state before back-propagation
            self.encoder.train()
            self.decoders[current_domain.name].train()
            self.domain_discriminator.train()

            # accumulate the prediction loss
            encoded_feature, _ = self.encoder(word_seq, char_seq)
            decode_loss = self.decoders[current_domain.name].loss(encoded_feature, true_tags)
            meta_train_pred_loss += decode_loss

            # accumulate the domain discrimination loss
            adversial_loss = self.domain_discriminator.loss(encoded_feature, domain_tag)
            meta_train_loss_adv += adversial_loss

        return meta_train_pred_loss, meta_train_loss_adv

    def meta_validate(self, encoder_old_param, encoder_new_param, valid_domains):
        if not isinstance(valid_domains, list):  # ensure valid_domains are iterable
            valid_domains = [valid_domains]

        # initialize
        meta_valid_pred_loss = 0.0
        meta_valid_loss_adv = 0.0

        # two encoders are temporary models
        new_encoder = CNN_BiGRU(self.word_size, self.word_emb_dim, self.alphabet_size, self.char_emb_dim,
                                self.hidden_size, word_pad_idx=self.word_size, char_pad_idx=self.alphabet_size,
                                is_freeze=False, cnn_total_num=100, dropout=0.2, pretrained_path="pre_trained.pt")
        new_encoder.load_state_dict(encoder_new_param)
        old_encoder = CNN_BiGRU(self.word_size, self.word_emb_dim, self.alphabet_size, self.char_emb_dim,
                                self.hidden_size, word_pad_idx=self.word_size, char_pad_idx=self.alphabet_size,
                                is_freeze=False, cnn_total_num=100, dropout=0.2, pretrained_path="pre_trained.pt")
        old_encoder.load_state_dict(encoder_old_param)

        new_encoder.train()
        old_encoder.train()

        for current_domain in valid_domains:
            selected_samples = random.sample(current_domain.train, self.batch_size)
            (word_seq, char_seq), true_tags, domain_tag = current_domain.process_data(selected_samples)

            # get the prediction loss
            encoded_feature_old, _ = old_encoder(word_seq, char_seq)
            decode_loss = self.decoders[current_domain.name].loss(encoded_feature_old, true_tags)
            meta_valid_pred_loss += decode_loss

            # get the domain discrimination loss
            encoded_feature_new, _ = new_encoder(word_seq, char_seq)
            adversiral_loss = self.domain_discriminator.loss(encoded_feature_new, domain_tag)
            meta_valid_loss_adv += adversiral_loss

        meta_valid_loss = meta_valid_pred_loss + meta_valid_loss_adv
        return meta_valid_loss

    def evaluate(self, domains):
        """
        evaluates the model performance in the domains (valid dataset)
        :param domains:
        :return: average pred_loss, macro f1
        """
        # make the models into eval state
        self.encoder.eval()
        for current_domain in self.train_domains:
            self.decoders[current_domain.name].eval()

        # start evaluation
        with torch.no_grad():
            pred_loss = 0  # total loss
            total_sample_num = 0
            gold_label = []
            prediction = []

            for domain in domains:
                print("Evaluating valid set of {}".format(domain.name))

                for start_id in range(0, len(domain.valid), self.batch_size):
                    end_id = min(start_id + self.batch_size, len(domain.valid))
                    selected_data = domain.valid[start_id: end_id]
                    total_sample_num += len(selected_data)
                    (word_seq, char_seq), true_tags, domain_tag = domain.process_data(selected_data)

                    feature, _ = self.encoder(word_seq, char_seq)

                    # get the training loss
                    score = self.decoders[domain.name].loss(feature, true_tags)
                    pred_loss += -score.item()
                    # get the prediction fixme: check
                    unpad_true_tag = [tag for sentence in true_tags.tolist() for tag in sentence if tag != -1]
                    gold_label.extend(unpad_true_tag)
                    real_entries_mask = true_tags != -1
                    pred_tag = self.decoders[domain.name].decode(feature, mask=real_entries_mask)
                    prediction.extend([x for y in pred_tag for x in y])
            pred_loss = pred_loss / total_sample_num
            f1 = f1_score(gold_label, prediction, average='macro')
        return pred_loss, f1
