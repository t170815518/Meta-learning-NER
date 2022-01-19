import argparse
import pickle
import random
import numpy as np
import torch

from Dataset import Dataset
from meta_NER_algorithm import Meta_NER_Trainer

parser = argparse.ArgumentParser(description='Meta-NER')
parser.add_argument('--random_seed', type=int, default=777)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epoch_num', type=int, default=70)
parser.add_argument('--inner_lr', type=float, default=1e-4)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--eval_interval', type=int, default=3)
args = parser.parse_args()

# set random seed
random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

TRAIN_DOMAIN_NAMES = ["conll2003", "wikigold", "wnut17"]
DEVICE = torch.device("cpu")
CORPUS_SIZE = 400000
ALPHABET_SIZE = 249

word2id = None
id2word = None
with open("word2id.pkl", "rb") as f:
    word2id = pickle.load(f)
with open("id2word.pkl", "rb") as f:
    id2word = pickle.load(f)

train_domains = []
for domain_name in TRAIN_DOMAIN_NAMES:
    dataset = Dataset(domain_name, word2id, id2word, DEVICE)
    train_domains.append(dataset)

# todo: fix the word_embedding lr
trainer = Meta_NER_Trainer(batch_size=args.batch_size, epoch_num=args.epoch_num, train_domains=train_domains,
                           inner_lr=args.inner_lr, lr=args.lr, word_emb_lr=args.inner_lr, weight_decay=None,
                           eval_interval=args.eval_interval, alphabet_size=ALPHABET_SIZE,
                           word_size=CORPUS_SIZE)
trainer.train()
