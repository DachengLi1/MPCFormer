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

from single_models import Bert, BertEmbeddings

# Inference arguments
class config():
   def __init__(self):
       self.batch_size = 8
       self.num_hidden_layers = 12
       self.hidden_size = 768
       self.intermediate_size = 3072
       self.sequence_length = 512
       self.max_position_embeddings = 512
       self.hidden_act = "relu"
       self.softmax_act = "softmax"
       self.layer_norm_eps = 1e-12
       self.num_attention_heads = 12
       self.vocab_size = 28996
       self.hidden_dropout_prob = 0.1
       self.attention_probs_dropout_prob = 0.1

config = config()
print(f"using model config: {config}")

# setup fake data for timing purpose
#commInit = crypten.communicator.get().get_communication_stats()
input_ids = F.one_hot(torch.randint(low=0, high=config.vocab_size, size=(config.batch_size, config.sequence_length)), config.vocab_size).float().cuda()

timing = defaultdict(float)
m = Bert(config, timing).cuda()

for i in range(10):
    #timing = defaultdict(float)
    m.reset_timing()
    time_s = time.time()
    # run a forward pass
    with torch.no_grad():
        m(input_ids)
    torch.cuda.synchronize()

    time_e = time.time()
    timing["total_time"] = (time_e - time_s)
    print(timing)
