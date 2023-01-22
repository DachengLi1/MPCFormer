from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def softmax_2relu(scores):
    dim = -1
    eps = 1e-12
    relu = torch.nn.functional.relu(scores)
    reduce_dim = scores.shape[dim]
    out = (relu + eps/reduce_dim) / (torch.sum(relu, dim=dim, keepdims=True)+eps)
    #print(torch.isnan(out).any(), out.shape)
    return out

class QuadActivation(nn.Module):
    def forward(self, input):
        return 0.125*torch.square(input) + 0.25*input + 0.5

def define_tsnet(name, num_class, act, cuda=True):
    if name == 'resnet20':
        net = resnet20(num_class=num_class, act=act)
    elif name == 'resnet110':
        net = resnet110(num_class=num_class)
    elif name == 'resnet32':
        net = resnet32(num_class=num_class, act=act)
    elif name == 'mcccnn8':
        net = mcccnn8(num_class=num_class, act=act)
    elif name == 'mcccnn8_layernorm':
        net = mcccnn8_layernorm(num_class=num_class)
    elif name == 'mcccnn8_poly':
        net = mcccnn8_poly(num_class=num_class)
    elif name == 'mcccnn8_poly_layernorm':
        net = mcccnn8_poly_layernorm(num_class=num_class)
    else:
        raise Exception('model name does not exist.')

    if cuda:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = torch.nn.DataParallel(net)

    return net


class resblock(nn.Module):
    def __init__(self, in_channels, out_channels, return_before_act, act="relu"):
        super(resblock, self).__init__()
        self.return_before_act = return_before_act
        self.downsample = (in_channels != out_channels)
        if self.downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            self.ds    = nn.Sequential(*[
                            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                            nn.BatchNorm2d(out_channels)
                            ])
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.ds    = None
        self.bn1   = nn.BatchNorm2d(out_channels)
        #self.relu  = QuadActivation() #nn.ReLU()
        #self.relu    = lambda x: 0.125*x**2 + 0.5*x + 0.25
        if act == "relu":
            self.act  = nn.ReLU()
        else:
            assert act == "quad"
            self.act = QuadActivation()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x

        pout = self.conv1(x) # pout: pre out before activation
        pout = self.bn1(pout)
        pout = self.act(pout)

        pout = self.conv2(pout)
        pout = self.bn2(pout)

        if self.downsample:
            residual = self.ds(x)

        pout += residual
        out  = self.act(pout)

        if not self.return_before_act:
            return out
        else:
            return pout, out

class mcccnn8(nn.Module):
    def __init__(self, num_class, act):
        super(mcccnn8, self).__init__()
        
        self.conv1   = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(96)
        self.conv2   = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2     = nn.BatchNorm2d(96)
        self.conv3   = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3     = nn.BatchNorm2d(96)
        self.conv4   = nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4     = nn.BatchNorm2d(192)
        self.conv5   = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5     = nn.BatchNorm2d(192)
        self.conv6   = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6     = nn.BatchNorm2d(192)
        self.conv7   = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7     = nn.BatchNorm2d(192)
        self.conv8   = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8     = nn.BatchNorm2d(192)

        self.pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(3072, num_class)
        if act == "relu":
            self.act    = nn.ReLU()
        else:
            self.act = QuadActivation()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.act(out)

        out = self.conv6(out)
        out = self.bn6(out)
        out = self.conv7(out)
        out = self.bn7(out)
        out = self.conv8(out)
        out = self.bn8(out)
        out = self.act(out)
        
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out  = self.fc(out)
        return None, None, None, None, None, out

class mcccnn8_layernorm(nn.Module):
    def __init__(self, num_class):
        super(mcccnn8_layernorm, self).__init__()
        
        self.conv1   = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1     = nn.LayerNorm([96,32,32])
        self.conv2   = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2     = nn.LayerNorm([96,32,32])
        self.conv3   = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3     = nn.LayerNorm([96,32,32])
        self.conv4   = nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4     = nn.LayerNorm([192,32,32])
        self.conv5   = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5     = nn.LayerNorm([192,32,32])
        self.conv6   = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6     = nn.LayerNorm([192,32,32])
        self.conv7   = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7     = nn.LayerNorm([192,32,32])
        self.conv8   = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8     = nn.LayerNorm([192,32,32])

        self.pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(3072, num_class)
        self.act    = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.act(out)

        out = self.conv6(out)
        out = self.bn6(out)
        out = self.conv7(out)
        out = self.bn7(out)
        out = self.conv8(out)
        out = self.bn8(out)
        out = self.act(out)
        
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out  = self.fc(out)
        return None, None, None, None, None, out

class mcccnn8_poly_layernorm(nn.Module):
    def __init__(self, num_class):
        super(mcccnn8_poly_layernorm, self).__init__()
        
        self.conv1   = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1     = nn.LayerNorm([96,32,32])
        self.conv2   = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2     = nn.LayerNorm([96,32,32])
        self.conv3   = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3     = nn.LayerNorm([96,32,32])
        self.conv4   = nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4     = nn.LayerNorm([192,32,32])
        self.conv5   = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5     = nn.LayerNorm([192,32,32])
        self.conv6   = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6     = nn.LayerNorm([192,32,32])
        self.conv7   = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7     = nn.LayerNorm([192,32,32])
        self.conv8   = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8     = nn.LayerNorm([192,32,32])

        self.pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(3072, num_class)
        self.act    = lambda x: 0.125*x**2 + 0.5*x + 0.25

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.act(out)

        out = self.conv6(out)
        out = self.bn6(out)
        out = self.conv7(out)
        out = self.bn7(out)
        out = self.conv8(out)
        out = self.bn8(out)
        out = self.act(out)
        
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out  = self.fc(out)
        return None, None, None, None, None, out

class resnet20(nn.Module):
    def __init__(self, num_class, act="relu"):
        super(resnet20, self).__init__()
        self.conv1   = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(16)
        #self.relu    = QuadActivation() #nn.ReLU()
        #self.relu    = lambda x: 0.125*x**2 + 0.5*x + 0.25
        if act == "relu":
            self.act    = nn.ReLU()
        elif act == "quad":
            self.act = QuadActivation()
        self.act_type = act
        self.res1 = self.make_layer(resblock, 2, 16, 16)
        self.res2 = self.make_layer(resblock, 2, 16, 32)
        self.res3 = self.make_layer(resblock, 2, 32, 64)

        self.avgpool = nn.AvgPool2d(8)
        self.fc      = nn.Linear(64, num_class)

        for m in self.modules():
            continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.num_class = num_class

    def make_layer(self, block, num, in_channels, out_channels): # num must >=2
        layers = [block(in_channels, out_channels, False, self.act_type)]
        for i in range(num-2):
            layers.append(block(out_channels, out_channels, False, self.act_type))
        layers.append(block(out_channels, out_channels, True, self.act_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        pstem = self.conv1(x) # pstem: pre stem before activation
        pstem = self.bn1(pstem)
        stem  = self.act(pstem)
        stem  = (pstem, stem)

        rb1 = self.res1(stem[1])
        rb2 = self.res2(rb1[1])
        rb3 = self.res3(rb2[1])

        feat = self.avgpool(rb3[1])
        feat = feat.view(feat.size(0), -1)
        #print(feat.shape)
        out  = self.fc(feat)

        return stem, rb1, rb2, rb3, feat, out

    def get_channel_num(self):
        return [16, 16, 32, 64, 64, self.num_class]

    def get_chw_num(self):
        return [(16, 32, 32),
                (16, 32, 32),
                (32, 16, 16),
                (64, 8 , 8 ),
                (64,),
                (self.num_class,)]

class resnet32(nn.Module):
    def __init__(self, num_class, act="relu"):
        super(resnet32, self).__init__()
        self.conv1   = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(16)
        #self.relu    = QuadActivation() #nn.ReLU()
        #self.relu    = lambda x: 0.125*x**2 + 0.5*x + 0.25
        if act == "relu":
            self.act    = nn.ReLU()
        elif act == "quad":
            self.act = QuadActivation()
        self.act_type = act
        self.res1 = self.make_layer(resblock, 5, 16, 16)
        self.res2 = self.make_layer(resblock, 5, 16, 32)
        self.res3 = self.make_layer(resblock, 5, 32, 64)

        self.avgpool = nn.AvgPool2d(8)
        self.fc      = nn.Linear(64, num_class)

        for m in self.modules():
            continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.num_class = num_class

    def make_layer(self, block, num, in_channels, out_channels): # num must >=2
        layers = [block(in_channels, out_channels, False, self.act_type)]
        for i in range(num-2):
            layers.append(block(out_channels, out_channels, False, self.act_type))
        layers.append(block(out_channels, out_channels, True, self.act_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        pstem = self.conv1(x) # pstem: pre stem before activation
        pstem = self.bn1(pstem)
        stem  = self.act(pstem)
        stem  = (pstem, stem)

        rb1 = self.res1(stem[1])
        rb2 = self.res2(rb1[1])
        rb3 = self.res3(rb2[1])

        feat = self.avgpool(rb3[1])
        feat = feat.view(feat.size(0), -1)
        #print(feat.shape)
        out  = self.fc(feat)

        return stem, rb1, rb2, rb3, feat, out

    def get_channel_num(self):
        return [16, 16, 32, 64, 64, self.num_class]

    def get_chw_num(self):
        return [(16, 32, 32),
                (16, 32, 32),
                (32, 16, 16),
                (64, 8 , 8 ),
                (64,),
                (self.num_class,)]

class resnet110(nn.Module):
    def __init__(self, num_class):
        super(resnet110, self).__init__()
        self.conv1   = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(16)
        self.relu    = nn.ReLU()

        self.res1 = self.make_layer(resblock, 18, 16, 16)
        self.res2 = self.make_layer(resblock, 18, 16, 32)
        self.res3 = self.make_layer(resblock, 18, 32, 64)

        self.avgpool = nn.AvgPool2d(8)
        self.fc      = nn.Linear(64, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.num_class = num_class

    def make_layer(self, block, num, in_channels, out_channels):  # num must >=2
        layers = [block(in_channels, out_channels, False)]
        for i in range(num-2):
            layers.append(block(out_channels, out_channels, False))
        layers.append(block(out_channels, out_channels, True))
        return nn.Sequential(*layers)

    def forward(self, x):
        pstem = self.conv1(x) # pstem: pre stem before activation
        pstem = self.bn1(pstem)
        stem  = self.relu(pstem)
        stem  = (pstem, stem)

        rb1 = self.res1(stem[1])
        rb2 = self.res2(rb1[1])
        rb3 = self.res3(rb2[1])

        feat = self.avgpool(rb3[1])
        feat = feat.view(feat.size(0), -1)
        out  = self.fc(feat)

        return stem, rb1, rb2, rb3, feat, out

    def get_channel_num(self):
        return [16, 16, 32, 64, 64, self.num_class]

    def get_chw_num(self):
        return [(16, 32, 32),
                (16, 32, 32),
                (32, 16, 16),
                (64, 8 , 8 ),
                (64,),
                (self.num_class,)]


def define_paraphraser(in_channels_t, k, use_bn, cuda=True):
    net = paraphraser(in_channels_t, k, use_bn)
    if cuda:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = torch.nn.DataParallel(net)

    return net


class paraphraser(nn.Module):
    def __init__(self, in_channels_t, k, use_bn=True):
        super(paraphraser, self).__init__()
        factor_channels = int(in_channels_t*k)
        self.encoder = nn.Sequential(*[
                nn.Conv2d(in_channels_t, in_channels_t, 3, 1, 1, bias=bool(1-use_bn)),
                nn.BatchNorm2d(in_channels_t) if use_bn else nn.Sequential(),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(in_channels_t, factor_channels, 3, 1, 1, bias=bool(1-use_bn)),
                nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(factor_channels, factor_channels, 3, 1, 1, bias=bool(1-use_bn)),
                nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
                nn.LeakyReLU(0.1, inplace=True),
            ])
        self.decoder = nn.Sequential(*[
                nn.ConvTranspose2d(factor_channels, factor_channels, 3, 1, 1, bias=bool(1-use_bn)),
                nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
                nn.LeakyReLU(0.1, inplace=True),
                nn.ConvTranspose2d(factor_channels, in_channels_t, 3, 1, 1, bias=bool(1-use_bn)),
                nn.BatchNorm2d(in_channels_t) if use_bn else nn.Sequential(),
                nn.LeakyReLU(0.1, inplace=True),
                nn.ConvTranspose2d(in_channels_t, in_channels_t, 3, 1, 1, bias=bool(1-use_bn)),
                nn.BatchNorm2d(in_channels_t) if use_bn else nn.Sequential(),
                nn.LeakyReLU(0.1, inplace=True),
            ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z   = self.encoder(x)
        out = self.decoder(z)
        return z, out


def define_translator(in_channels_s, in_channels_t, k, use_bn=True, cuda=True):
    net = translator(in_channels_s, in_channels_t, k, use_bn)
    if cuda:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = torch.nn.DataParallel(net)

    return net


class translator(nn.Module):
    def __init__(self, in_channels_s, in_channels_t, k, use_bn=True):
        super(translator, self).__init__()
        factor_channels = int(in_channels_t*k)
        self.encoder = nn.Sequential(*[
                nn.Conv2d(in_channels_s, in_channels_s, 3, 1, 1, bias=bool(1-use_bn)),
                nn.BatchNorm2d(in_channels_s) if use_bn else nn.Sequential(),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(in_channels_s, factor_channels, 3, 1, 1, bias=bool(1-use_bn)),
                nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(factor_channels, factor_channels, 3, 1, 1, bias=bool(1-use_bn)),
                nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
                nn.LeakyReLU(0.1, inplace=True),
            ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z   = self.encoder(x)
        return z

