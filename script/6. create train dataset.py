import os
import random
from multiprocessing import Pool

import anndata
import cooler
import numpy as np
import pandas as pd
from tqdm import tqdm


map_info = pd.read_csv('/home/micl/workspace/lmh_data/sclab/map_result.csv', sep='\t', index_col=0)

# hic
cools_path = '/home/micl/workspace/lmh_data/Lee2019/Human_single_cell_10kb_cool'
hic = anndata.read_h5ad("/home/micl/workspace/lmh_data/sclab/hic_result.h5ad")
# rna
rna = anndata.read_h5ad("/home/micl/workspace/lmh_data/sclab/rna_result.h5ad")
rna = rna[map_info.index, :]

rna_marker_gene = pd.read_csv('/home/micl/workspace/lmh_data/sclab/rna_marker_gene.csv', sep='\t', index_col=0)
_tmp = rna.var.loc[rna_marker_gene.index]
_tmp = _tmp[_tmp['chromEnd'] > _tmp['chromStart'] + 50000]
rna_marker_gene = rna_marker_gene.loc[_tmp.index]

threads = 16

def mat2array(mat):
    return mat[np.triu_indices_from(mat, k=0)]

def get_cooler(hic_name):
    file_name = '{}_10kb_contacts.cool'.format(hic_name)
    return hic_name, cooler.Cooler(os.path.join(cools_path, file_name))
with Pool(threads) as p:
    _coolers = dict(p.map(get_cooler, hic.obs_names))

def catch_matrix(_c, chrom: str, start: int, end: int, strand: str):
    if strand == '+':
        start -= 5000
    else:
        end += 5000

    # contact = _c.matrix(balance=False, as_pixels=True, join=True).fetch('{}:{}-{}'.format(chrom, start, end))
    mat = _c.matrix(balance=False).fetch('{}:{}-{}'.format(chrom, start, end))
    return mat

def catch_location(_args):
    hic_name, gene_name = _args
    _gene_info = rna.var.loc[gene_name]
    start, end = int(_gene_info['chromStart']), int(_gene_info['chromEnd'])
    mat = catch_matrix(_coolers[hic_name], _gene_info['chrom'], start, end, _gene_info['strand'])
    return gene_name, mat2array(mat).astype(np.uint16)

def catch_locations(hic_names, gene_names):
    _contacts = pd.DataFrame()
    for hic_name in tqdm(hic_names):
        with Pool(threads) as p:
            _args = [(hic_name, gene_name) for gene_name in gene_names]
            _contact = pd.DataFrame([dict(p.map(catch_location, _args))], index=[hic_name])
        _contacts = pd.concat([_contacts, _contact], axis=0)
    return _contacts
contacts = catch_locations(hic.obs_names, rna_marker_gene.index.tolist())


_dataset = []
for scRNA, row in tqdm(map_info.iterrows(), total = map_info.shape[0]):
    _dataset.append({
        'scRNA': rna[scRNA, :].layers["counts"][0].astype(np.int32),
        'scRNA_head': rna.var_names,
        'scHiC': contacts.loc[row['scHiC']].to_dict(),
        'cell_type': row['cell_type']
    })

# fake_dataset = []
# for _gene in rna_marker_gene.index:


# Randomly generate train dataset and test dataset
random.seed(0)
_indexs = random.sample(range(0, len(_dataset)), int(len(_dataset)*0.1))
eval_dataset = [_dataset[_index] for _index in _indexs]
_indexs = list(set(range(0, len(_dataset))) - set(_indexs))
train_dataset = [_dataset[_index] for _index in _indexs]

np.save('/home/micl/workspace/lmh_data/sclab/train_dataset.npy', train_dataset)
np.save('/home/micl/workspace/lmh_data/sclab/eval_dataset.npy', eval_dataset)