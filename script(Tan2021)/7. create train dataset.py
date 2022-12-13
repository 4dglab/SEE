import os
import random
from multiprocessing import Pool

import anndata
import cooler
import numpy as np
import pandas as pd
from tqdm import tqdm

root_dir = '/lmh_data/data/sclab'
threads = 8
random.seed(0)

map_info = pd.read_csv(os.path.join(root_dir, 'sclab', 'mouse', 'map_result.csv'), sep='\t', index_col=0)
# hic
cools_path = os.path.join(root_dir, 'Tan2021', 'GSE162511_10kb_cool')
hic = anndata.read_h5ad(os.path.join(root_dir, 'sclab', 'mouse', 'hic_result.h5ad'))
# rna
rna = anndata.read_h5ad(os.path.join(root_dir, 'sclab', 'mouse', 'rna_result.h5ad'))
rna_marker_gene = pd.read_csv(os.path.join(root_dir, 'sclab', 'mouse', 'rna_marker_gene.csv'), sep='\t', index_col=0)
_tmp = rna.var.loc[rna_marker_gene.index]
_tmp = _tmp[_tmp['chromEnd'] > _tmp['chromStart'] + 50000]
rna_marker_gene = rna_marker_gene.loc[_tmp.index]


def get_cooler(hic_name):
    for _, _, file_names in os.walk(cools_path):
        for file_name in file_names:
            if hic_name in file_name:
                return hic_name, cooler.Cooler(os.path.join(cools_path, file_name))
    raise Exception('The file could not be found')
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

def mat2array(mat):
    return mat[np.triu_indices_from(mat, k=0)]

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
        'cell_type': row['cell_type'],
        'identity': 'truth',
    })

def get_corr_matrix():
    hic_corr, rna_corr = dict(), dict()
    for cell_type in map_info['cell_type'].unique():
        _hic_names = hic[hic.obs['cell_type']==cell_type].obs_names
        _rna_names = rna[rna.obs['cell_type']==cell_type].obs_names

        _contacts = contacts.loc[_hic_names]
        _hic = np.array([np.concatenate(i, axis=0) for i in _contacts.values])
        _hic = pd.DataFrame(data=_hic.T, columns=_contacts.index)
        _rna = pd.DataFrame(data=rna[_rna_names].X.T, columns=rna[_rna_names].obs_names)

        hic_corr[cell_type], rna_corr[cell_type] = _hic.corr(), _rna.corr()
    return hic_corr, rna_corr
hic_corr, rna_corr = get_corr_matrix()

def find_best_RNA(_name, cell_type):
    best_rna_names = rna_corr[cell_type].loc[_name].nlargest().index
    _data = np.array(rna[best_rna_names].layers["counts"].mean(axis=0)).astype(np.int32)
    return _data

def find_best_HiC(_name, cell_type):
    best_hic_names = hic_corr[cell_type].loc[_name].nlargest().index
    _data = {}
    for column, row in contacts.loc[best_hic_names].iteritems():
        _data[column] = np.ceil(np.vstack(row.values).mean(axis=0)).astype(np.int16)
    _data = pd.DataFrame([_data])
    return _data

cell_type_count = map_info.groupby('cell_type').count()
for cell_type, row in cell_type_count.iterrows():
    for i in range(min(cell_type_count.max()[0] - row['scHiC'], int(row['scHiC']/2))):
        random_index = random.randint(0, row['scHiC'] - 1)
        _case = map_info[map_info['cell_type']==cell_type].iloc[random_index]
        best_rna, best_hic = find_best_RNA(_case.name, cell_type), find_best_HiC(_case['scHiC'], cell_type)
        
        _dataset.append({
            'scRNA': best_rna,
            'scRNA_head': rna.var_names,
            'scHiC': best_hic.loc[0].to_dict(),
            'cell_type': cell_type,
            'identity': 'fake',
        })


# Randomly generate train dataset and test dataset
_indexs = random.sample(range(0, len(_dataset)), int(len(_dataset)*0.1))
eval_dataset = [_dataset[_index] for _index in _indexs]
_indexs = list(set(range(0, len(_dataset))) - set(_indexs))
train_dataset = [_dataset[_index] for _index in _indexs]

np.save(os.path.join(root_dir, 'sclab', 'mouse', 'train_dataset.npy'), train_dataset)
np.save(os.path.join(root_dir, 'sclab', 'mouse', 'eval_dataset.npy'), eval_dataset)