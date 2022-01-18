import random
from collections import OrderedDict
import torch
from torch import optim


# debug data
total_sample_size = 50000
domain_num = 5


class Meta_NER_Trainer:
    """
    The wrapper class of solving Domain-adaptation NER with Meta-NER.
    The entry method is train().
    """
    def __init__(self, batch_size: int, epoch_num: int, train_domains, inner_lr: float, lr: float, word_emb_lr: float,
                 weight_decay: float, eval_interval: int):
        assert batch_size > 0
        assert epoch_num > 0
        assert inner_lr > 0.0
        assert lr > 0.0
        assert eval_interval > 0
        assert word_emb_lr > 0

        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.train_domains = train_domains
        self.inner_lr = inner_lr
        self.eval_interval = eval_interval

        # initialize the network components
        self.encoder = None
        self.decoders = {}
        self.domain_discriminator = None
        for domain in self.train_domains:
            self.decoders[domain.name] = None
        # initialize the optimizer
        self.encoder_optimizer = optim.Adam([{'params': self.encoder.wordrep.word_embed.parameters(),
                                              'lr': word_emb_lr, 'weight_decay': 0},
                                             {'params': self.encoder.base_params}],
                                            lr=lr, weight_decay=weight_decay)
        self.decoder_optimizers = {}
        for domain in self.train_domains:
            self.decoder_optimizers[domain.name] = optim.Adam([{'params': self.decoders[domain.name].parameters()}],
                                                              lr=lr, weight_decay=weight_decay)

    def train(self):
        """
        The core method of the class
        """
        # to approximate the iterations and epoch index
        iteration_per_epoch = int(total_sample_size / (self.batch_size * domain_num))
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
            decoder_weights = []
            for name in self.train_domains:
                decoder_weights[name] = OrderedDict(
                        (name, param) for (name, param) in self.decoders[name].named_parameters())

            domain_discriminator_weights = OrderedDict((name, param) for (name, param)
                                                       in self.domain_discriminator.named_parameters())
            # take the derivative
            grad_encoder_old = torch.autograd.grad(meta_pred_loss, encoder_weights.values(), retain_graph=True)
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
            for name in meta_train_domains:  # also update the encoder for meta-train domains
                grads_decode_final[name] = torch.autograd.grad(meta_pred_loss + meta_adversial_loss,
                                                               decoder_weights[name].values(), retain_graph=True)
            for name in meta_val_domain:
                grads_decode_final[name] = torch.autograd.grad(meta_valid_loss,
                                                               decoder_weights[name].values(), retain_graph=True)
            for initial_param, final_grad in zip(self.encoder.parameters(), grads_encoder_final):
                initial_param.grad = final_grad

            for name in self.train_domains:
                for initial_param, final_grad in zip(self.decoders[name].parameters(), grads_decode_final[name]):
                    initial_param.grad = final_grad

            for initial_param, final_grad in zip(self.domain_discriminator.parameters(), grads_discriminator_final):
                initial_param.grad = final_grad

            # clip_norm
            self.encoder_optimizer.step()
            for name in self.train_domains:
                self.decoder_optimizers[name].step()
            self.domain_discriminator.step()

            # check-point
            if i % self.eval_interval == 0 and i != 0:
                performance_str, f1_score = self.evaluate(self.meta_train_domains)
                print("==============meta_train===========")
                print(performance_str)
                if f1_score > best_train_f1:
                    best_train_f1 = f1_score

                performance_str, f1_score = self.evaluate(self.dev_data)
                print("==============meta_train===========")
                print(performance_str)
                if f1_score > best_dev_f1:
                    # todo: save the best model
                    best_dev_f1 = f1_score

    def meta_train(self, meta_train_domains):
        meta_train_pred_loss = 0.0
        meta_train_loss_adv = 0.0

        for current_domain in meta_train_domains:
            input_feature, true_tags, domain_tag = current_domain.sample(self.batch_size)

            # set the model's state before back-propagation
            self.encoder.train()
            self.decoders[current_domain.name].train()
            self.domain_discriminator.train()

            # accumulate the prediction loss
            encoded_feature = self.encoder(input_feature)
            decode_loss = self.decoders[current_domain.name].crf_neglog_loss(encoded_feature, true_tags)
            meta_train_pred_loss += decode_loss

            # accumulate the domain discrimination loss
            adversial_loss = self.domain_discriminator.loss(encoded_feature, domain_tag)
            meta_train_loss_adv += adversial_loss
        meta_train_loss = meta

    def meta_validate(self, encoder_old, encoder_new, valid_domains):
        if len(valid_domains) == 1:  # ensure valid_domains are iterable
            valid_domains = [valid_domains]

        # initialize
        meta_valid_pred_loss = 0.0
        meta_valid_loss_adv = 0.0

        # two encoders are temporary models
        new_encoder = None
        old_encoder = None

        new_encoder.train()
        old_encoder.train()

        for current_domain in valid_domains:
            input_feature, true_tags, domain_tag = current_domain.sample(self.batch_size)

            # get the prediction loss
            encoded_feature_old = old_encoder(input_feature)
            decode_loss = self.decoders[current_domain.name].crf_neglog_loss(encoded_feature_old, true_tags)
            meta_valid_pred_loss += decode_loss

            # get the domain discrimination loss
            encoded_feature_new = new_encoder(input_feature)
            adversial_loss = self.domain_discriminator.loss(encoded_feature_new, domain_tag)
            meta_valid_loss_adv += adversial_loss

        meta_valid_loss = meta_valid_pred_loss + meta_valid_loss_adv
        return meta_valid_loss

    def evaluate(self, domains):
        performance_str = ""
        f1_score = 0

        # make the models into eval state
        self.encoder.eval()
        for current_domain in self.train_domains:
            self.encoders[current_domain].eval()

        # start evaluation
        for domain in domains:
            for input_feature, true_tags, domain_tag in domain:
                features = self.encoder(input_feature)
                score = self.decoders[domain.name].decode(features, true_tags)

        return performance_str, f1_score
