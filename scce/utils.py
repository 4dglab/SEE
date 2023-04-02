import os
import numpy as np


def mkdir(out_dir):
    if os.path.splitext(out_dir)[-1] != '':
        out_dir = os.path.dirname(out_dir)
    if not os.path.isdir(out_dir):
        print(f'Making directory: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)

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

