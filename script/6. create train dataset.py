import os
import random

import anndata
import cooler
import numpy as np
import pandas as pd
from tqdm import tqdm


map_info = pd.read_csv('/home/micl/workspace/lmh_data/sclab/map_result.csv', sep='\t', index_col=0)

# hic
cools_path = '/home/micl/workspace/lmh_data/Lee2019/Human_single_cell_10kb_cool'
# rna
rna = anndata.read_h5ad("/home/micl/workspace/lmh_data/sclab/rna_result.h5ad")
rna = rna[map_info.index, :]

rna_marker_gene = pd.read_csv('/home/micl/workspace/lmh_data/sclab/rna_marker_gene.csv', sep='\t', index_col=0)
_tmp = rna.var.loc[rna_marker_gene.index]
_tmp = _tmp[_tmp['chromEnd'] > _tmp['chromStart'] + 50000]
rna_marker_gene = rna_marker_gene.loc[_tmp.index]

def catch_location(_c, chrom: str, start: int, end: int, strand: str):
    if strand == '+':
        start -= 5000
    else:
        end += 5000

    # contact = _c.matrix(balance=False, as_pixels=True, join=True).fetch('{}:{}-{}'.format(chrom, start, end))
    mat = _c.matrix(balance=False).fetch('{}:{}-{}'.format(chrom, start, end))
    return mat

_dataset = []
for scRNA, row in tqdm(map_info.iterrows(), total = map_info.shape[0]):
    file_name = '{}_10kb_contacts.cool'.format(row['scHiC'])
    _c = cooler.Cooler(os.path.join(cools_path, file_name))
    _dict = {}
    for _gene in rna_marker_gene.index:
        _gene_info = rna.var.loc[_gene]
        start, end = int(_gene_info['chromStart']), int(_gene_info['chromEnd'])
        mat = catch_location(_c, _gene_info['chrom'], start, end, _gene_info['strand'])
        _dict[_gene] = mat.astype(np.uint16)
    _dataset.append({
        'scRNA': rna[scRNA, :].layers["counts"][0].astype(np.int32),
        'scRNA_head': rna.var_names,
        'scHiC': _dict,
        'cell_type': row['cell_type']
    })

random.seed(0)
_indexs = random.sample(range(0, len(_dataset)), int(len(_dataset)*0.1))
eval_dataset = [_dataset[_index] for _index in _indexs]
_indexs = list(set(range(0, len(_dataset))) - set(_indexs))
train_dataset = [_dataset[_index] for _index in _indexs]

np.save('/home/micl/workspace/lmh_data/sclab/train_dataset.npy', train_dataset)
np.save('/home/micl/workspace/lmh_data/sclab/eval_dataset.npy', eval_dataset)