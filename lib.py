from easydl import *
import numpy as np

def reverse_sigmoid(y):
    return torch.log(y / (1.0 - y + 1e-10) + 1e-10)


def normalize_weight(x, cut=0, expand=False):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val + 1e-10) / (max_val - min_val + 1e-10)
    if expand:
        x = x / torch.mean(x)
        # x = torch.where(x >= cut, x, torch.zeros_like(x))
    return x.detach()


def l2_norm(input, dim=1):
    norm = torch.norm(input,dim=dim,keepdim=True)
    output = torch.div(input, norm)
    return output


def seed_everything(seed=1234):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
