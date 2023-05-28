import os
import logging

import numpy as np
import torch
import torch.distributed as dist


def mkdir(out_dir):
    if os.path.splitext(out_dir)[-1] != '':
        out_dir = os.path.dirname(out_dir)
    if not os.path.isdir(out_dir):
        print(f'Making directory: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def init_dist(backend='nccl', **kwargs):
    dist.init_process_group(backend='nccl')

    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return local_rank, device


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def mat2array(mat):
    return mat[np.triu_indices_from(mat, k=0)]

def array2mat(array):
    _len, a = len(array), 0
    while _len:
        a += 1
        _len -= a
    _len = a

    mat, a = np.zeros((_len, _len)), 0
    for i in range(_len):
        mat[i, i:] = array[a:a+_len-i]
        a += _len - i
    return mat + np.triu(mat, k=1).T