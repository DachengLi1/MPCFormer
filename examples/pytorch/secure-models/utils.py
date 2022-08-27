import torch

import crypten
import crypten.communicator as comm
import crypten.nn as cnn

def encrypt_tensor(input):
    """Encrypt data tensor for multi-party setting"""
    # get rank of current process
    rank = comm.get().get_rank()
    # get world size
    world_size = comm.get().get_world_size()
    assert world_size  == 2
    
    # assumes party 1 is the actual data provider
    src_id = 1

    if rank == src_id:
        input_upd = input.cuda()
    else:
        input_upd = torch.empty(input.size()).cuda()
    private_input = crypten.cryptensor(input_upd, src=src_id)
#    print(private_input)
    return private_input

def encrypt_model(model, modelFunc, config, dummy_input):
    rank = comm.get().get_rank()
    
    # assumes party 0 is the actual model provider
    if rank == 0:
        model_upd = model.cuda()
    else:
        if isinstance(config, tuple):
            model_upd = modelFunc(config[0], config[1]).cuda()
        else:
            model_upd = modelFunc(config).cuda()

    private_model = model_upd.encrypt(src=0)
    return private_model


class softmax_2RELU(cnn.Module):
    def __init__(self, dim):
        super().__init__()
        self.func = cnn.ReLU()
        self.dim = dim

    def forward(self, x):
        func_x = self.func(x)
        return func_x / func_x.sum(keepdim=True, dim=self.dim)

class softmax_2QUAD(cnn.Module):
    def __init__(self, norm, dim):
        super().__init__()
        self.dim = dim
        self.norm = norm
    
    def forward(self, x):
        a, b, c, d = x.size()
        quad = x#self.norm(x)
        quad = (quad*quad + quad + 1)
        return quad / quad.sum(dim=self.dim, keepdims=True)

class activation_quad(cnn.Module):
    def __init__(self):
        super().__init__()
        self.first_coef = torch.tensor([0.125]).item()
        self.second_coef = torch.tensor([0.5]).item()
        self.third_coef = torch.tensor([0.25]).item()
        self.pow = torch.tensor([2]).item()
     
    def forward(self, x):
        return self.first_coef*x*x + self.second_coef*x + self.third_coef
        #return x*x

