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

from network import mcccnn8

# Inference arguments
class config():
   def __init__(self):
       self.batch_size = 1
       self.num_class = 10
       self.act = "relu"
       self.softmax_act = "softmax"

config = config()
print(f"using model config: {config}")

# 2PC setting
rank = sys.argv[1]
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(2)
os.environ["MASTER_ADDR"] = "172.31.12.12"
os.environ["MASTER_PORT"] = "29501"
os.environ["RENDEZVOUS"] = "env://"

crypten.init()
cfg.communicator.verbose = True

# setup fake data for timing purpose
commInit = crypten.communicator.get().get_communication_stats()
input = torch.randn(config.batch_size, 3, 32, 32).cuda()

timing = defaultdict(float)

m = mcccnn8(config, timing)
model = encrypt_model(m, mcccnn8, (config, timing), input).eval()

# encrpy inputs
input = encrypt_tensor(input)

for i in range(10):
    #m.reset_timing()
    time_s = time.time()
    # run a forward pass
    with crypten.no_grad():
        model(input)

    time_e = time.time()
    timing["total_time"] = (time_e - time_s)
    print(timing)
