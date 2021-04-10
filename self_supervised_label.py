import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
from itertools import combinations
import copy
import backbone as backbone
import configs
from methods.protonet import euclidean_dist


def class_balanced_ss(num_lab = 2):
    max_val_tup = torch.max(score, 1)
    argmax_val = max_val_tup[1]
    max_val = max_val_tup[0]
    max_val_idx = [(i, val) for i,val in enumerate(max_val)]
    total_indices = []
    for i in range(n_way): ##class balanced relabelling of query samples
        offset = i * n_query
        max_val_class = [(j, val) for j, val in max_val_idx if argmax_val[j] == i]
        if len(max_val_class) > num_lab:
        max_val_class.sort(key = lambda x:x[1], reverse = True)
        max_val_class = max_val_class[:num_lab]
        total_indices.extend([j for j,val in max_val_class])

    final_class = [argmax_val[idx].cpu().numpy() for idx in total_indices]

    return final_class, total_indices
