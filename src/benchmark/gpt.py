import time
import math

import torch
import torch.nn.functional as F

import crypten
import crypten.nn as cnn
import crypten.communicator as comm
from crypten.common.functions import maximum

from utils import softmax_2RELU, softmax_2QUAD, activation_quad, activation_newGeLU, encrypt_tensor

class gpt(cnn.Module):
    def __init__(self, config, timing):
        super(gpt, self).__init__()
        self.config = config

        # No need to init weight for timing purpose
        self.embeddings = gptEmbeddings(config, timing)
        self.encoder = cnn.ModuleList([gptLayer(config, timing) for _ in range(config.num_hidden_layers)])
        self.lm_head = cnn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.smax = cnn.Softmax(dim=-1)
        self.cat = cnn.Concat(dimension=1)
        self.timing = timing
   
    def reset_timing(self):
        for k,v in self.timing.items():
            self.timing[k] = 0
 
    def forward(self, input_ids, past_list):
        output = self.embeddings(input_ids)
        for layer_id, layer in enumerate(self.encoder):
            # pass in a past key/value of shape [[b, s, h], [b, s, h]] !!not tuple, it will get deep copied..!!
            if len(past_list[layer_id]) == 0:
                print("input to layer None")
            else:
                print("input to layer size: ", past_list[layer_id][0].shape, past_list[layer_id][1].shape)
            #output, past = layer(output, past_list[layer_id])
            output = layer(output, past_list[layer_id])
            #past_list[layer_id].append()
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        output = self.lm_head(output)
        comm1 = comm.get().get_communication_stats()
        t1 = time.time()
        self.timing["LinearTime"] += (t1-t0)
        self.timing["LinearCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["LinearCommByte"] += (comm1["bytes"] - comm0["bytes"])
        self.timing["lmHeadTime"] += (t1-t0)
        self.timing["lmHeadCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["lmHeadCommByte"] += (comm1["bytes"] - comm0["bytes"])
        return output#, past

    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,s,v)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        generation_time = {}
        past_list = [[] for _ in range(self.config.num_hidden_layers)]
        generation_stage = False
        for token_id in range(max_new_tokens):
            b, s, _ = idx.shape
            time_s = time.time()
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.max_position_embeddings else idx[:, -self.config.max_position_embeddings:,:]
            # forward the model to get the logits for the index in the sequence
            #print(idx_cond.shape)
            if not generation_stage:
                logits = self(idx_cond, past_list)
                generation_stage = True
            else:
                logits = self(idx_cond[:, -1:, :], past_list)
            #print("logit shape: ", logits.shape)
            # pluck the logits at the final step and scale by desired temperature
            t0 = time.time()
            comm0 = comm.get().get_communication_stats()
            logits = logits[:, -1:, :] / temperature
            #print("logits shape: ", logits.shape)
            # optionally crop the logits to only the top k options
            #if top_k is not None:
            #    v, _ = torch.topk(logits, top_k)
            #    logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = self.smax(logits)
            # either sample from the distribution or take the most likely element
            #if do_sample:
            #    idx_next = torch.multinomial(probs, num_samples=1)
            #else:
            #print("prob size: ", probs.shape)
            idx_next = maximum.argmax(probs, dim=-1)
            #print("next idx:", idx_next.shape)
            # append sampled index to the running sequence and continue
            #idx_next = F.one_hot(idx_next, self.config.vocab_size).cuda()
            #idx_next = encrypt_tensor(idx_next)
            #print("pre-cat size: ",idx.shape, idx_next.shape)
            idx = self.cat([idx, idx_next])
            comm1 = comm.get().get_communication_stats()
            t1 = time.time()
            time_e = time.time()
            generation_time.update({(b, s): time_e - time_s})
            self.timing["GenerateOtherTime"] += (t1-t0)
            self.timing["GenerateOtherCommTime"] += (comm1["time"] - comm0["time"])
            self.timing["GenerateOtherCommByte"] += (comm1["bytes"] - comm0["bytes"])
            print(generation_time)
        return idx

class gptEmbeddings(cnn.Module):
    def __init__(self, config, timing):
        super(gptEmbeddings, self).__init__()
        # save memory
        self.pruneFactor = 250
        self.tokenSubDim = config.vocab_size // self.pruneFactor
        self.lastTokenDim = config.vocab_size - (self.pruneFactor - 1) * self.tokenSubDim
        self.moduleList = []

        for _ in range(self.pruneFactor - 1):
            ll = cnn.Linear(self.tokenSubDim, config.hidden_size)
            self.moduleList.append(ll)

        self.moduleList.append(cnn.Linear(self.lastTokenDim, config.hidden_size))

        self.wpe = cnn.Linear(config.max_position_embeddings, config.hidden_size)
        #print(config.hidden_size)
        self.LayerNorm = cnn.BatchNorm2d(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = cnn.Dropout(config.hidden_dropout_prob)
        self.config = config
        self.timing = timing

    def reset_timing(self):
        for k,v in self.timing.items():
            self.timing[k] = 0
    
    def cuda(self, device=None):
        super(gptEmbeddings, self).cuda(device=device)

        for i in range(len(self.moduleList)):
            self.moduleList[i].cuda(device=device)
        self.wpe.cuda(device=device)
        return self

    def encrypt(self, mode=True, src=0):
        super(gptEmbeddings, self).encrypt(mode=mode, src=src)

        for i in range(len(self.moduleList)):
            self.moduleList[i].encrypt(mode=mode, src=src)
        self.wpe.encrypt(mode=mode, src=src)
        return self

    def forward(self, input_ids):
        embeddings = None
        
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        for i, ll in enumerate(self.moduleList):
            #print(ll.weight.shape)
            if i != (len(self.moduleList) - 1):
            #   print(input_ids[:, :, i * self.tokenSubDim : (i + 1) * self.tokenSubDim].shape)
                res = ll(input_ids[:, :, i * self.tokenSubDim : (i + 1) * self.tokenSubDim])
            else:
                res = ll(
                    input_ids[
                        :,:,
                        i * self.tokenSubDim : i * self.tokenSubDim + self.lastTokenDim
                    ]
                )

            embeddings = res if embeddings is None else (embeddings + res)

        t1 = time.time()
        comm1 = comm.get().get_communication_stats()
        self.timing["EmbedTime"] += (t1-t0)
        self.timing["EmbedCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["ËmbedCommByte"] += (comm1["bytes"] - comm0["bytes"])
        #print("benchmarking embed: ", input_ids.shape, t1-t0)

        position_embeddings = (self.wpe.weight[:,:input_ids.shape[1]]).transpose(0,1)
     #   print(position_embeddings.shape, self.position_embeddings.weight.shape)
        position_embeddings = position_embeddings.repeat(input_ids.shape[0],1,1)
     #   print(position_embeddings.shape, embeddings.shape)
        embeddings += position_embeddings

        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        orig_size = embeddings.size()
        embeddings = embeddings.view(-1, self.config.hidden_size)
        embeddings = self.LayerNorm(embeddings).view(orig_size)
        t1 = time.time()
        comm1 = comm.get().get_communication_stats()
        self.timing["NormTime"] += (t1-t0)
        self.timing["NormCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["NormCommByte"] += (comm1["bytes"] - comm0["bytes"])
        embeddings = self.dropout(embeddings)
        return embeddings

class gptLayer(cnn.Module):
    def __init__(self, config, timing):
        super(gptLayer, self).__init__()
        self.config = config
        self.attention = gptAttention(config, timing)
        self.intermediate = gptIntermediate(config, timing)
        self.output = gptOutput(config, timing)
        self.config = config
        self.timing = timing
 
    def reset_timing(self):
        for k,v in self.timing.items():
            self.timing[k] = 0
 
    def forward(self, hidden_states, past):
        #attention_output, past = self.attention(hidden_states, past)
        #print("debug copy before: ", past)
        attention_output = self.attention(hidden_states, past)
        #print("debug copy after: ", past)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output#, past
        
class gptAttention(cnn.Module):
    def __init__(self, config, timing):
        super(gptAttention, self).__init__()
        self.self = gptSelfAttention(config, timing)
        self.output = gptSelfOutput(config, timing)
    
    def reset_timing(self):
        for k,v in self.timing.items():
            self.timing[k] = 0
    
    def forward(self, hidden_states, past):
        #self_output, past = self.self(hidden_states, past)
        self_output = self.self(hidden_states, past)
        attention_output = self.output(self_output, hidden_states)
        return attention_output#, past

class gptSelfAttention(cnn.Module):
    def __init__(self, config, timing):
        super(gptSelfAttention, self).__init__()
        
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = self.hidden_size // self.num_attention_heads

        self.query = cnn.Linear(self.hidden_size, self.hidden_size)
        self.key = cnn.Linear(self.hidden_size, self.hidden_size)
        self.value = cnn.Linear(self.hidden_size, self.hidden_size)

        self.cat = cnn.Concat(dimension=-2)
        # TODO: implement causal mask
        #self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                             .view(1, 1, config.block_size, config.block_size))
        self.dropout = cnn.Dropout(config.attention_probs_dropout_prob)
        if config.softmax_act == "softmax":
            self.smax = cnn.Softmax(dim=-1)
        elif config.softmax_act == "softmax_2RELU":
            self.smax = softmax_2RELU(dim=-1)
        elif config.softmax_act == "softmax_2QUAD":
            self.norm = cnn.BatchNorm2d(config.hidden_size, eps=config.layer_norm_eps)
            self.smax = softmax_2QUAD(self.norm, dim=-1)
        else:
            raise ValueError(f"softmax type {config.softmax_act} not implemented.")
        self.timing = timing
    
    def reset_timing(self):
        for k,v in self.timing.items():
            self.timing[k] = 0

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, past):
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        print("key shape:", key_layer.shape)
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        if len(past) != 0:
            past_key, past_value = past
            print("cat debug: ", past_key.shape, key_layer.shape )
            key_layer = self.cat([past_key, key_layer])
            value_layer = self.cat([past_value, value_layer])
            past[0] = key_layer
            past[1] = value_layer
        else:
            # update past
            past.append(key_layer)
            past.append(value_layer)        
           
        attention_scores = query_layer.matmul(key_layer.transpose(-1, -2))
        #print(attention_scores.shape)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # TODO: implement mask
        # attention_scores = attention_scores.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        t1 = time.time()
        comm1 = comm.get().get_communication_stats()
        self.timing["LinearTime"] += (t1 - t0)
        self.timing["LinearCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["LinearCommByte"] += (comm1["bytes"] - comm0["bytes"])
        
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        #print("smax operands: ", attention_scores.shape)
        attention_probs = self.smax(attention_scores)
        t1 = time.time()
        comm1 = comm.get().get_communication_stats()
        self.timing["SoftmaxTime"] += (t1 - t0)
        self.timing["SoftmaxCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["SoftmaxCommByte"] += (comm1["bytes"] - comm0["bytes"])

        attention_probs = self.dropout(attention_probs)
        
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        context_layer = attention_probs.matmul(value_layer)
        t1 = time.time()
        comm1 = comm.get().get_communication_stats()
        self.timing["LinearTime"] += (t1 - t0)
        self.timing["LinearCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["LinearCommByte"] += (comm1["bytes"] - comm0["bytes"])

        context_layer = context_layer.permute(0, 2, 1, 3)#.contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        #print("debug shapes after attention: ", context_layer.shape, key_layer.shape, value_layer.shape)        
        return context_layer#, (key_layer, value_layer)

class gptSelfOutput(cnn.Module):
    def __init__(self, config, timing):
        super(gptSelfOutput, self).__init__()
        self.dense = cnn.Linear(config.hidden_size, config.hidden_size)
        # using batchnorm here, crypten has not implemented LayerNorm
        self.LayerNorm = cnn.BatchNorm2d(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = cnn.Dropout(config.hidden_dropout_prob)
        self.timing = timing
        self.config = config
    
    def reset_timing(self):
        for k,v in self.timing.items():
            self.timing[k] = 0

    def forward(self, hidden_states, input_tensor):
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        hidden_states = self.dense(hidden_states)
        t1 = time.time()
        comm1 = comm.get().get_communication_stats()
        self.timing["LinearTime"] += (t1 - t0)
        self.timing["LinearCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["LinearCommByte"] += (comm1["bytes"] - comm0["bytes"])
        
        hidden_states = self.dropout(hidden_states)
        # residual connection here
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        orig_size = hidden_states.size()
        hidden_states = hidden_states + input_tensor
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        hidden_states = self.LayerNorm(hidden_states).view(orig_size)
        t1 = time.time()
        comm1 = comm.get().get_communication_stats()
        self.timing["NormTime"] += (t1 - t0)
        self.timing["NormCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["NormCommByte"] += (comm1["bytes"] - comm0["bytes"])
        return hidden_states

class gptIntermediate(cnn.Module):
    def __init__(self, config, timing):
        super(gptIntermediate, self).__init__()
        self.dense = cnn.Linear(config.hidden_size, config.intermediate_size)
        if config.hidden_act == "relu":
            self.intermediate_act_fn = cnn.ReLU()
        elif config.hidden_act == "quad":
            self.intermediate_act_fn = activation_quad()
        elif config.hidden_act == "newGeLU":
            self.intermediate_act_fn = activation_newGeLU()
        else:
            raise ValueError(f"activation type {config.hidden_act} not implemented")
        self.timing = timing

    def reset_timing(self):
        for k,v in self.timing.items():
            self.timing[k] = 0
    
    def forward(self, hidden_states):
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        hidden_states = self.dense(hidden_states)
        t1 = time.time()
        comm1 = comm.get().get_communication_stats()
        self.timing["LinearTime"] += (t1 - t0)
        self.timing["LinearCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["LinearCommByte"] += (comm1["bytes"] - comm0["bytes"])
        
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        hidden_states = self.intermediate_act_fn(hidden_states)
        t1 = time.time()
        comm1 = comm.get().get_communication_stats()
        self.timing["ActTime"] += (t1 - t0)
        self.timing["ActCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["ActCommByte"] += (comm1["bytes"] - comm0["bytes"])
        return hidden_states


class gptOutput(cnn.Module):
    def __init__(self, config, timing):
        super(gptOutput, self).__init__()
        self.dense = cnn.Linear(config.intermediate_size, config.hidden_size)
        # using batchnorm here, crypten has not implemented LayerNorm
        self.LayerNorm = cnn.BatchNorm2d(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = cnn.Dropout(config.hidden_dropout_prob)
        self.timing = timing
        self.config = config
    
    def reset_timing(self):
        for k,v in self.timing.items():
            self.timing[k] = 0

    def forward(self, hidden_states, input_tensor):
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        hidden_states = self.dense(hidden_states)
        t1 = time.time()
        comm1 = comm.get().get_communication_stats()
        self.timing["LinearTime"] += (t1 - t0)
        self.timing["LinearCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["LinearCommByte"] += (comm1["bytes"] - comm0["bytes"])
        hidden_states = self.dropout(hidden_states)
        # residual connection
        t0 = time.time()
        comm0 = comm.get().get_communication_stats()
        orig_size = hidden_states.size()
        hidden_states = hidden_states + input_tensor
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        hidden_states = self.LayerNorm(hidden_states).view(orig_size)
        t1 = time.time()
        comm1 = comm.get().get_communication_stats()
        self.timing["NormTime"] += (t1 - t0)
        self.timing["NormCommTime"] += (comm1["time"] - comm0["time"])
        self.timing["NormCommByte"] += (comm1["bytes"] - comm0["bytes"])
        return hidden_states
