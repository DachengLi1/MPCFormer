from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

import crypten.nn as cnn
from utils import activation_quad, softmax_2RELU

class resblock(nn.Module):
    def __init__(self, in_channels, out_channels, return_before_act):
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
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x

        pout = self.conv1(x) # pout: pre out before activation
        pout = self.bn1(pout)
        pout = self.relu(pout)

        pout = self.conv2(pout)
        pout = self.bn2(pout)

        if self.downsample:
            residual = self.ds(x)

        pout += residual
        out  = self.relu(pout)

        if not self.return_before_act:
            return out
        else:
            return pout, out

class mcccnn8(cnn.Module):
    def __init__(self, config, timing):
        super(mcccnn8, self).__init__()
        
        self.conv1   = cnn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1     = cnn.BatchNorm1d(96)
        self.conv2   = cnn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2     = cnn.BatchNorm1d(96)
        self.conv3   = cnn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3     = cnn.BatchNorm1d(96)
        self.conv4   = cnn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4     = cnn.BatchNorm1d(192)
        self.conv5   = cnn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5     = cnn.BatchNorm1d(192)
        self.conv6   = cnn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6     = cnn.BatchNorm1d(192)
        self.conv7   = cnn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7     = cnn.BatchNorm1d(192)
        self.conv8   = cnn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8     = cnn.BatchNorm1d(192)

        self.pool = cnn.AvgPool2d(8)
        self.fc = cnn.Linear(3072, config.num_class)
        if config.act == "relu":
            self.act  = cnn.ReLU()
        elif config.act == "quad":
            self.act = activation_quad()
         
        if config.softmax_act == "softmax":
            self.smax = cnn.Softmax(dim=-1)
        elif config.softmax_act == "softmax_2RELU":
            self.smax = softmax_2RELU(dim=-1)
   
    def forward(self, x):
        print(x.shape)
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
        out = self.smax(out)
        return out

class mcccnn8_poly(nn.Module):
    def __init__(self, num_class):
        super(mcccnn8_poly, self).__init__()
        
        self.conv1   = nn.Conv2d(3, 300, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(300)
        self.conv2   = nn.Conv2d(300, 300, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2     = nn.BatchNorm2d(300)
        self.conv3   = nn.Conv2d(300, 300, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3     = nn.BatchNorm2d(300)
        self.conv4   = nn.Conv2d(300, 600, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4     = nn.BatchNorm2d(600)
        self.conv5   = nn.Conv2d(600, 600, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5     = nn.BatchNorm2d(600)
        self.conv6   = nn.Conv2d(600, 600, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6     = nn.BatchNorm2d(600)
        self.conv7   = nn.Conv2d(600, 600, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7     = nn.BatchNorm2d(600)
        self.conv8   = nn.Conv2d(600, 600, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8     = nn.BatchNorm2d(600)

        self.pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(9600, num_class)
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
    def __init__(self, num_class):
        super(resnet20, self).__init__()
        self.conv1   = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(16)
        self.relu    = nn.ReLU()

        self.res1 = self.make_layer(resblock, 3, 16, 16)
        self.res2 = self.make_layer(resblock, 3, 16, 32)
        self.res3 = self.make_layer(resblock, 3, 32, 64)

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

    def make_layer(self, block, num, in_channels, out_channels): # num must >=2
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

# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        #self.encoder = nn.Embedding(ntoken, ninp)
        self.encoder = nn.Linear(ntoken, ninp, bias=False)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)
