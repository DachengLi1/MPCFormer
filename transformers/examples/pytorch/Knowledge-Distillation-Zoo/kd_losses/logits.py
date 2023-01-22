from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


class Logits(nn.Module):
	'''
	Do Deep Nets Really Need to be Deep?
	http://papers.nips.cc/paper/5484-do-deep-nets-really-need-to-be-deep.pdf
	'''
	def __init__(self):
		super(Logits, self).__init__()

	def forward(self, out_s, out_t):
		loss = F.mse_loss(out_s, out_t)

		return loss
