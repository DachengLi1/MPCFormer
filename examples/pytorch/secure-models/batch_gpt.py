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

from gpt import gpt, gptEmbeddings, gptSelfAttention, gptLayer, gptIntermediate, gptOutput, gptSelfOutput

# Inference arguments
class config():
   def __init__(self):
       self.batch_size = 1
       self.num_hidden_layers = 12
       self.hidden_size = 5120
       self.intermediate_size = self.hidden_size * 4
       self.sequence_length = 16
       self.max_position_embeddings = 1024
       self.hidden_act = "newGeLU"
       self.softmax_act = "softmax"
       self.layer_norm_eps = 1e-12
       self.num_attention_heads = 40
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
#input_ids = F.one_hot(torch.randint(low=0, high=config.vocab_size, size=(config.batch_size, config.sequence_length)), config.vocab_size).float().cuda()
#inputs = torch.randn(config.batch_size, config.sequence_length, config.hidden_size).cuda()
inputs = torch.randn(config.batch_size, config.sequence_length, config.hidden_size).cuda()
#inputs_2 = torch.randn(config.batch_size, config.sequence_length, config.intermediate_size).cuda()
inputs_2 = torch.randn(config.batch_size, config.sequence_length, config.hidden_size).cuda()

timing = defaultdict(float)

#m = gpt(config, timing)
#m = gptEmbeddings(config, timing)
#m = gptSelfAttention(config, timing)
#m = gptLayer(config, timing)
#m = gptIntermediate(config, timing)
#m = gptOutput(config, timing)
m = gptSelfOutput(config, timing)
#model = encrypt_model(m, gptEmbeddings, (config, timing), input_ids).eval()
#model = encrypt_model(m, gptSelfAttention, (config, timing), inputs).eval()
#model = encrypt_model(m, gptLayer, (config, timing), inputs).eval()
#model = encrypt_model(m, gptIntermediate, (config, timing), inputs).eval()
#model = encrypt_model(m, gptOutput, (config, timing), inputs).eval()
model = encrypt_model(m, gptSelfOutput, (config, timing), inputs).eval()

# encrpy inputs
#input_ids = encrypt_tensor(input_ids)
input_ids = encrypt_tensor(inputs)
input_2 = encrypt_tensor(inputs_2)

import numpy as np
time_dict_list = {}
for i in range(1):
    m.reset_timing()
    time_s = time.time()
    # run a forward pass
    with crypten.no_grad():
        #model.generate(input_ids, 32)
        model(input_2, input_ids)
        #model(input_ids)
        #model(input_ids, [])
    time_e = time.time()
    timing["total_time"] = (time_e - time_s)
    print(timing)
    if len(time_dict_list.keys()) == 0:
        for k, v in timing.items():
             time_dict_list[k] = [v]
    else:
        for k, v in timing.items():
             time_dict_list[k].append(v)

for k, v in time_dict_list.items():
    arr = np.asarray(time_dict_list[k])
    time_dict_list[k] = (np.mean(arr), np.std(arr))
print(time_dict_list)

