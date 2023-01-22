import math
import time

import torch
import torch.nn.functional as F
import torch.nn as cnn

from utils import softmax_2RELU, activation_quad

class Bert(cnn.Module):
    def __init__(self, config, timing):
        super(Bert, self).__init__()
        self.config = config

        self.embeddings = BertEmbeddings(config, timing)
        self.encoder = cnn.ModuleList([BertLayer(config, timing) for _ in range(config.num_hidden_layers)])
        self.timing = timing        

    def reset_timing(self):
        for k, v in self.timing.items():
            self.timing[k] = 0
    
    def forward(self, input_ids):
        output = self.embeddings(input_ids)
        for _, layer in enumerate(self.encoder):
            output = layer(output)
        return output

class BertEmbeddings(cnn.Module):
    def __init__(self, config, timing):
        super(BertEmbeddings, self).__init__()
        # save memory
        self.pruneFactor = 250
        self.tokenSubDim = config.vocab_size // self.pruneFactor
        self.lastTokenDim = config.vocab_size - (self.pruneFactor - 1) * self.tokenSubDim
        self.moduleList = []

        for _ in range(self.pruneFactor - 1):
            ll = cnn.Linear(self.tokenSubDim, config.hidden_size)
            self.moduleList.append(ll)

        self.moduleList.append(cnn.Linear(self.lastTokenDim, config.hidden_size))
        self.moduleList = cnn.ModuleList(self.moduleList)

        self.position_embeddings = cnn.Linear(config.max_position_embeddings, config.hidden_size)
#        print(config.hidden_size)
        self.LayerNorm = cnn.BatchNorm1d(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = cnn.Dropout(config.hidden_dropout_prob)
        self.config = config
        self.timing = timing

    def cuda(self, device=None):
        super(BertEmbeddings, self).cuda(device=device)

        for i in range(len(self.moduleList)):
            self.moduleList[i].cuda(device=device)
        return self

    def forward(self, input_ids):
        embeddings = None
        
        t0 = time.time()
        for i, ll in enumerate(self.moduleList):
            #print(ll.weight.shape)
 #           print(input_ids, ll)
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
        self.timing["EmbedTime"] += (t1-t0)

        position_embeddings = (self.position_embeddings.weight[:,:input_ids.shape[1]]).transpose(0,1)
     #   print(position_embeddings.shape, self.position_embeddings.weight.shape)
        position_embeddings = position_embeddings.repeat(input_ids.shape[0],1,1)
     #   print(position_embeddings.shape, embeddings.shape)
        embeddings += position_embeddings

        t0 = time.time()
        orig_size = embeddings.size()
        embeddings = embeddings.view(-1, self.config.hidden_size)
        embeddings = self.LayerNorm(embeddings).view(orig_size)
        t1 = time.time()
        self.timing["NormTime"] += (t1-t0)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertLayer(cnn.Module):
    def __init__(self, config, timing):
        super(BertLayer, self).__init__()
        self.config = config
        self.attention = BertAttention(config, timing)
        self.intermediate = BertIntermediate(config, timing)
        self.output = BertOutput(config, timing)
        self.config = config
 
    def forward(self, hidden_states):
        attention_output = self.attention(hidden_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
        
class BertAttention(cnn.Module):
    def __init__(self, config, timing):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config, timing)
        self.output = BertSelfOutput(config, timing)
    
    def forward(self, hidden_states):
        self_output = self.self(hidden_states)
        attention_output = self.output(self_output, hidden_states)
        return attention_output 

class BertSelfAttention(cnn.Module):
    def __init__(self, config, timing):
        super(BertSelfAttention, self).__init__()
        
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = self.hidden_size // self.num_attention_heads

        self.query = cnn.Linear(self.hidden_size, self.hidden_size)
        self.key = cnn.Linear(self.hidden_size, self.hidden_size)
        self.value = cnn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = cnn.Dropout(config.attention_probs_dropout_prob)
        if config.softmax_act == "softmax":
            self.smax = cnn.Softmax(dim=-1)
        elif config.softmax_act == "softmax_2RELU":
            self.smax = softmax_2RELU(dim=-1)
        else:
            raise ValueError(f"softmax type {config.softmax_act} not implemented.")
        self.timing = timing

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        t0 = time.time()
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        attention_scores = query_layer.matmul(key_layer.transpose(-1, -2))
        #print(attention_scores.shape)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        t1 = time.time()
        self.timing["LinearTime"] += (t1 - t0)
        
        t0 = time.time()
        attention_probs = self.smax(attention_scores)
        t1 = time.time()
        self.timing["SoftmaxTime"] += (t1 - t0)

        attention_probs = self.dropout(attention_probs)
        
        t0 = time.time()
        context_layer = attention_probs.matmul(value_layer)
        t1 = time.time()
        self.timing["LinearTime"] += (t1 - t0)

        context_layer = context_layer.permute(0, 2, 1, 3)#.contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)
        
        return context_layer

class BertSelfOutput(cnn.Module):
    def __init__(self, config, timing):
        super(BertSelfOutput, self).__init__()
        self.dense = cnn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = cnn.BatchNorm1d(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = cnn.Dropout(config.hidden_dropout_prob)
        self.timing = timing
        self.config = config

    def forward(self, hidden_states, input_tensor):
        t0 = time.time()
        hidden_states = self.dense(hidden_states)
        t1 = time.time()
        self.timing["LinearTime"] += (t1 - t0)
        
        hidden_states = self.dropout(hidden_states)
        # residual connection here
        t0 = time.time()
        orig_size = hidden_states.size()
        hidden_states = hidden_states + input_tensor
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        hidden_states = self.LayerNorm(hidden_states).view(orig_size)
        t1 = time.time()
        self.timing["NormTime"] += (t1 - t0)
        return hidden_states

class BertIntermediate(cnn.Module):
    def __init__(self, config, timing):
        super(BertIntermediate, self).__init__()
        self.dense = cnn.Linear(config.hidden_size, config.intermediate_size)
        if config.hidden_act == "relu":
            self.intermediate_act_fn = cnn.ReLU()
        elif config.hidden_act == "quad":
            self.intermediate_act_fn = activation_quad()
        else:
            raise ValueError(f"activation type {config.hidden_act} not implemented")
        self.timing = timing

    def forward(self, hidden_states):
        t0 = time.time()
        hidden_states = self.dense(hidden_states)
        t1 = time.time()
        self.timing["LinearTime"] += (t1 - t0)
        
        t0 = time.time()
        hidden_states = self.intermediate_act_fn(hidden_states)
        t1 = time.time()
        self.timing["ActTime"] += (t1 - t0)
        return hidden_states


class BertOutput(cnn.Module):
    def __init__(self, config, timing):
        super(BertOutput, self).__init__()
        self.dense = cnn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = cnn.BatchNorm1d(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = cnn.Dropout(config.hidden_dropout_prob)
        self.timing = timing
        self.config = config

    def forward(self, hidden_states, input_tensor):
        t0 = time.time()
        hidden_states = self.dense(hidden_states)
        t1 = time.time()
        self.timing["LinearTime"] += (t1 - t0)
        hidden_states = self.dropout(hidden_states)
        # residual connection
        t0 = time.time()
        orig_size = hidden_states.size()
        hidden_states = hidden_states + input_tensor
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        hidden_states = self.LayerNorm(hidden_states).view(orig_size)
        t1 = time.time()
        self.timing["NormTime"] += (t1 - t0)
        return hidden_states
