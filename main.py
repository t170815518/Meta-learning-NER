"""
Entry script
"""


import argparse
import pickle
import random
import numpy as np
import torch

from Dataset import Dataset
from meta_NER_algorithm import Meta_NER_Trainer


TRAIN_DOMAIN_NAMES = ["conll2003", "wikigold", "wnut17", "ontonotes"]
DEVICE = torch.device("cpu")
# fixme: remember to change the magic numbers when the train domains change
CORPUS_SIZE = 400000
ALPHABET_SIZE = 269

# parse the argument
parser = argparse.ArgumentParser(description='Meta-NER')
parser.add_argument('--random_seed', type=int, default=777)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epoch_num', type=int, default=70)
parser.add_argument('--inner_lr', type=float, default=1e-4)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--alpha', type=float, default=1e-4)
parser.add_argument('--eval_interval', type=int, default=50)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--word_emb_lr', type=float, default=1e-4)
parser.add_argument('--word_emb_dim', type=int, default=300)
parser.add_argument('--char_emb_dim', type=int, default=100)
parser.add_argument('--hidden_size', type=int, default=128)
args = parser.parse_args()

# set random seed
random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

word2id = None
id2word = None
with open("word2id.pkl", "rb") as f:  # the files should be pre-processed
    word2id = pickle.load(f)
with open("id2word.pkl", "rb") as f:
    id2word = pickle.load(f)

train_domains = []
for domain_name in TRAIN_DOMAIN_NAMES:
    dataset = Dataset(domain_name, word2id, id2word, DEVICE)
    train_domains.append(dataset)

# todo: fix the word_embedding lr
trainer = Meta_NER_Trainer(batch_size=args.batch_size, train_domains=train_domains, device=DEVICE,
                           word_size=CORPUS_SIZE, word_emb_dim=args.word_emb_dim, alphabet_size=ALPHABET_SIZE,
                           char_emb_dim=args.char_emb_dim, lr=args.lr, word_emb_lr=args.word_emb_lr,
                           weight_decay=args.weight_decay, is_lr_decay=True,
                           total_train_size=Dataset.total_train_sample_size, epoch_num=args.epoch_num,
                           inner_lr=args.inner_lr, eval_interval=args.eval_interval, hidden_size=args.hidden_size)
trainer.train()
