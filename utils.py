import torch
import numpy as np
import math

import subprocess as sp
import os

def get_gpu_memory():
  _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

  ACCEPTABLE_AVAILABLE_MEMORY = 1024
  COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
  memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
  memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
  print(memory_free_values)
  return memory_free_values

mem = get_gpu_memory()

max_i = np.argmax(mem)

cuda_val = "cuda:" + str(max_i)
device = torch.device(cuda_val if torch.cuda.is_available() else "cpu")

def adjust_learning_rate(optimizer, epoch, lr=0.01, step1=30, step2=60, step3=90):
    """Sets the learning rate to the initial LR decayed by 10 every X epochs"""
    if epoch >= step3:
        lr = lr * 0.001
    elif epoch >= step2:
        lr = lr * 0.01
    elif epoch >= step1:
        lr = lr * 0.1
    else:
        lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0      
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def one_hot(y, num_class):         
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)

def sparsity(cl_data_file):
    class_list = cl_data_file.keys()
    cl_sparsity = []
    for cl in class_list:
        cl_sparsity.append(np.mean([np.sum(x!=0) for x in cl_data_file[cl] ])  ) 

    return np.mean(cl_sparsity) 

