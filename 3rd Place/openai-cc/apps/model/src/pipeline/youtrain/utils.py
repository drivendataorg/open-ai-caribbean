import random
import torch
import numpy as np
import yaml
import os

def set_global_seeds(i):
    torch.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    random.seed(i)
    np.random.seed(i)


def get_config(path):
    with open(path, 'r') as stream:
        config = yaml.load(stream)
    return config

def get_version(filename):
    n = filename.count('_')
    if n == 0:
        return filename
    elif n==1:
        return filename.split('_')[-1]
    elif n==2:
        return filename.split('_')[-2]
    else:
        print('Cant find version')

def get_last_save(path):
    if os.path.exists(path):
        files = [get_version('.'.join(f.split('.')[:-1])) for f in os.listdir(path) if '.pt' in f]
        numbers = []
        for f in files:
            try:
                numbers.append(int(f))
            except: pass
        if len(numbers) > 0:
            return max(numbers)
        else:
            return 0
    else:
        return 0


