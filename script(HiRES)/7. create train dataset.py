import os
import random
from multiprocessing import Pool

import anndata
import cooler
import numpy as np
import pandas as pd
import scglue
from tqdm import tqdm

root_dir = "/lmh_data/data/sclab"
threads = 8
random.seed(0)

# hic
cools_path = os.path.join(root_dir, "GSE223917", "cool")
hic = anndata.read_h5ad(os.path.join(root_dir, "GSE223917", "scHiC.h5ad"))
# rna
rna = anndata.read_h5ad(os.path.join(root_dir, "GSE223917", "scRNA.h5ad"))
rna_marker_gene = ['Pax6', 'Sox2', 'Sox4', 'Sox5', 'Hes6', 'Sox11', 'Bcl11b']
scglue.data.get_gene_annotation(
    rna, gtf="/lmh_data/data/sclab/GSE223917/gencode.vM23.annotation.gtf",
    gtf_by="gene_name"
)

# filter marker gene
_tmp = rna.var.loc[rna_marker_gene]
_tmp = _tmp[_tmp["chromEnd"] > _tmp["chromStart"] + 50000]
rna_marker_gene = _tmp.index.tolist()


def get_cooler(hic_name):
    for _, _, file_names in os.walk(cools_path):
        for file_name in file_names:
            if hic_name in file_name:
                return hic_name, cooler.Cooler(os.path.join(cools_path, file_name))
    raise Exception("The file could not be found")


with Pool(threads) as p:
    _coolers = dict(p.map(get_cooler, hic.obs_names))


def catch_matrix(_c, chrom: str, start: int, end: int, strand: str):
    if strand == "+":
        start -= 5000
    else:
        end += 5000

    # contact = _c.matrix(balance=False, as_pixels=True, join=True).fetch('{}:{}-{}'.format(chrom, start, end))
    mat = _c.matrix(balance=False).fetch("{}:{}-{}".format(chrom, start, end))
    return mat


def mat2array(mat):
    return mat[np.triu_indices_from(mat, k=0)]


def catch_location(_args):
    hic_name, gene_name = _args
    _gene_info = rna.var.loc[gene_name]
    start, end = int(_gene_info["chromStart"]), int(_gene_info["chromEnd"])
    mat = catch_matrix(
        _coolers[hic_name], _gene_info["chrom"], start, end, _gene_info["strand"]
    )
    return gene_name, mat2array(mat).astype(np.uint16)


def catch_locations(hic_names, gene_names):
    _contacts = pd.DataFrame()
    for hic_name in tqdm(hic_names):
        with Pool(threads) as p:
            _args = [(hic_name, gene_name) for gene_name in gene_names]
            _contact = pd.DataFrame(
                [dict(p.map(catch_location, _args))], index=[hic_name]
            )
        _contacts = pd.concat([_contacts, _contact], axis=0)
    return _contacts


contacts = catch_locations(hic.obs_names, rna_marker_gene)

_dataset = []
for sample_name, row in tqdm(rna.obs.iterrows(), total=rna.obs.shape[0]):
    _dataset.append(
        {
            "scRNA": rna[sample_name, :].X[0].astype(np.int32),
            "scRNA_head": rna.var_names,
            "scHiC": contacts.loc['{}.cool'.format(sample_name)].to_dict(),
            "cell_type": row["cell_type"],
            "identity": "truth",
        }
    )


# Randomly generate train dataset and test dataset
_indexs = random.sample(range(0, len(_dataset)), int(len(_dataset) * 0.1))
eval_dataset = [_dataset[_index] for _index in _indexs]
_indexs = list(set(range(0, len(_dataset))) - set(_indexs))
train_dataset = [_dataset[_index] for _index in _indexs]

np.save(os.path.join(root_dir, "GSE223917", "train_dataset.npy"), train_dataset)
np.save(os.path.join(root_dir, "GSE223917", "eval_dataset.npy"), eval_dataset)
