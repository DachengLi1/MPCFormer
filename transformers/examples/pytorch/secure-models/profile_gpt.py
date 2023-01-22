import sys
import os
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
#from transformers import AutoConfig, BertForSequenceClassificationWrapper

import crypten
import crypten.communicator as comm
from crypten.config import cfg
from utils import encrypt_tensor, encrypt_model

from gpt import gpt

# Inference arguments
class config():
   def __init__(self):
       self.batch_size = 1
       self.num_hidden_layers = 12
       self.hidden_size = 768
       self.intermediate_size = self.hidden_size * 4
       self.sequence_length = 16
       self.max_position_embeddings = 1024
       #self.hidden_act = "newGeLU"
       #self.softmax_act = "softmax"
       self.hidden_act = "quad"
       self.softmax_act = "softmax_2QUAD"
       self.layer_norm_eps = 1e-12
       self.num_attention_heads = 12
       self.vocab_size = 50257
       self.hidden_dropout_prob = 0.1
       self.attention_probs_dropout_prob = 0.1

config = config()
print(f"using model config: {config}")

# 2PC setting
rank = sys.argv[1]
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(2)
os.environ["MASTER_ADDR"] = "10.117.1.21"
os.environ["MASTER_PORT"] = "29500"
os.environ["RENDEZVOUS"] = "env://"

crypten.init()
cfg.communicator.verbose = True

# setup fake data for timing purpose
commInit = crypten.communicator.get().get_communication_stats()
input_ids = F.one_hot(torch.randint(low=0, high=config.vocab_size, size=(config.batch_size, config.sequence_length)), config.vocab_size).float().cuda()

timing = defaultdict(float)

m = gpt(config, timing)
model = encrypt_model(m, gpt, (config, timing), input_ids).eval()

# encrpy inputs
input_ids = encrypt_tensor(input_ids)

for i in range(1):
    m.reset_timing()
    time_s = time.time()
    # run a forward pass
    with crypten.no_grad():
        model.generate(input_ids, 32)

    time_e = time.time()
    timing["total_time"] = (time_e - time_s)
    print(timing)
