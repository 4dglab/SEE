import os
import random
from multiprocessing import Pool

import anndata
import click
import cooler
import numpy as np
import pandas as pd
from cooltools import insulation
from tqdm import tqdm


def get_cooler(_args):
    hic_name, cools_path = _args
    for _, _, file_names in os.walk(cools_path):
        for file_name in file_names:
            if hic_name in file_name:
                return hic_name, cooler.Cooler(os.path.join(cools_path, file_name))
    raise Exception("The file could not be found")

def catch_matrix(_c, chrom: str, start: int, end: int):
    mat = _c.matrix(balance=False).fetch("{}:{}-{}".format(chrom, start, end))
    return mat

def mat2array(mat):
    return mat[np.triu_indices_from(mat, k=0)]

def catch_location(_args):
    clr, chrom, chromStart, chromEnd = _args
    mat = catch_matrix(clr, chrom, chromStart, chromEnd)
    return "{}_{}_{}".format(chrom, chromStart, chromEnd), mat2array(mat).astype(np.uint16)

def catch_locations(clrs, hic_names, locations, threads):
    _contacts = pd.DataFrame()
    for hic_name in tqdm(hic_names):
        with Pool(threads) as p:
            _args = [(clrs[hic_name], location[0], location[1], location[2]) for location in locations]
            _contact = pd.DataFrame([dict(p.map(catch_location, _args))], index=[hic_name])
        _contacts = pd.concat([_contacts, _contact], axis=0)
    return _contacts

def get_random_locations(boundary, number=12):
    _boundary = boundary[boundary["level"]=="Strong"].reset_index(drop=True)
    domain_locations = []
    while len(domain_locations) < number:
        index = random.sample(_boundary.index.tolist(), 1)[0]
        if _boundary.iloc[index].chrom != _boundary.iloc[index+1].chrom:
            continue
        if (_boundary.iloc[index+1].start - _boundary.iloc[index].start) < 200000:
            continue
        domain_locations.append((
            _boundary.iloc[index].chrom,
            _boundary.iloc[index].start,
            _boundary.iloc[index+1].start
        ))
    
    return domain_locations

def get_corr_matrix(map_info, hic, rna, contacts):
    hic_corr, rna_corr = dict(), dict()
    for cell_type in map_info["cell_type"].unique():
        _hic_names = hic[hic.obs["cell_type"]==cell_type].obs_names
        _rna_names = rna[rna.obs["cell_type"]==cell_type].obs_names

        _contacts = contacts.loc[_hic_names]
        _hic = np.array([np.concatenate(i, axis=0) for i in _contacts.values])
        _hic = pd.DataFrame(data=_hic.T, columns=_contacts.index)
        _rna = pd.DataFrame(data=rna[_rna_names].X.T, columns=rna[_rna_names].obs_names)

        hic_corr[cell_type], rna_corr[cell_type] = _hic.corr(), _rna.corr()
    return hic_corr, rna_corr

def find_best_RNA(rna, rna_corr, _name, cell_type):
    best_rna_names = rna_corr[cell_type].loc[_name].nlargest().index
    _data = np.array(rna[best_rna_names].layers["counts"].mean(axis=0)).astype(np.int32)
    return _data

def find_best_HiC(contacts, hic_corr, _name, cell_type):
    best_hic_names = hic_corr[cell_type].loc[_name].nlargest().index
    _data = {}
    for column, row in contacts.loc[best_hic_names].iteritems():
        _data[column] = np.ceil(np.vstack(row.values).mean(axis=0)).astype(np.int16)
    _data = pd.DataFrame([_data])
    return _data

@click.group()
def cli():
    pass

@cli.command(short_help="Generate bulk hic based on single cell hic files.")
@click.option("--cools_path", type=str, default="/data/liminghong/sclab/Lee2019/Human_single_cell_10kb_cool", help="Path to the folder containing cool files.")
@click.option("--hic_file", type=str, default="/data/liminghong/sclab/sclab/hic_result.h5ad", help="Path to hic_result.h5ad.")
@click.option("--output_file", type=str, default="/data/liminghong/sclab/sclab/random_test/bulk.cool", help="Path to save dataset files.")
@click.option("--threads", type=int, default=8, help="Number of threads.")
def generate_bulk_hic(cools_path, hic_file, output_file, threads):
    hic = anndata.read_h5ad(hic_file)

    with Pool(threads) as p:
        _args = [(hic_name, cools_path) for hic_name in hic.obs_names]
        clrs = dict(p.map(get_cooler, _args))

    pixels = [clr.pixels()[:].set_index(["bin1_id", "bin2_id"]) for clr in tqdm(clrs.values())]
    pixels = pd.concat(pixels)
    pixels = pixels.groupby(pixels.index).sum()
    pixels.index = pd.MultiIndex.from_tuples(pixels.index)
    pixels = pixels.reset_index()
    pixels.rename(columns={"level_0": "bin1_id", "level_1": "bin2_id"}, inplace=True)

    cooler.create_cooler(
        cool_uri=output_file,
        bins=next(iter(clrs.values())).bins()[:],
        pixels=pixels,
        triucheck=False,
    )

@cli.command(short_help="Generate mask matrix based on bed files.")
@click.option("--cool_path", type=str, default="/data/liminghong/sclab/sclab/random_test/bulk.cool", help="Path to the bulk hic file.")
@click.option("--output_file", type=str, default="/data/liminghong/sclab/sclab/random_test/boundary.bed", help="Path to save dataset files.")
def get_boudary(cool_path, output_file):
    clr = cooler.Cooler(cool_path)
    resolution = 10000
    windows = [3*resolution, 5*resolution, 10*resolution, 25*resolution]
    insulation_table = insulation(clr, windows, clr_weight_name=None, ignore_diags=2, verbose=True)

    boundaries = insulation_table[~np.isnan(insulation_table[f"boundary_strength_{windows[0]}"])]
    # boundaries = boundaries[boundaries[boundaries.filter(regex='^is_boundary_').columns].any(axis=1)]
    weak_boundaries = boundaries[~boundaries[f"is_boundary_{windows[0]}"]]
    strong_boundaries = boundaries[boundaries[f"is_boundary_{windows[0]}"]]
    weak_boundaries.loc[:, "level"], strong_boundaries.loc[:, "level"] = "Weak", "Strong"
    
    result = pd.concat([weak_boundaries, strong_boundaries])
    result = result[["chrom", "start", "end", "level"]]
    result = result.sort_values(by=["chrom", "start"]).reset_index(drop=True)
    result.to_csv(output_file, sep="\t", index=False, header=False)

@cli.command(short_help="Generate dataset.")
@click.option("--map_file", type=str, default="/data/liminghong/sclab/sclab/map_result.csv", help="Path to map_result.csv.")
@click.option("--cools_path", type=str, default="/data/liminghong/sclab/Lee2019/Human_single_cell_10kb_cool", help="Path to the folder containing cool files.")
@click.option("--hic_file", type=str, default="/data/liminghong/sclab/sclab/hic_result.h5ad", help="Path to hic_result.h5ad.")
@click.option("--rna_file", type=str, default="/data/liminghong/sclab/sclab/rna_result.h5ad", help="Path to rna_result.h5ad.")
@click.option("--boundary_file", type=str, default="/data/liminghong/sclab/sclab/random_test/boundary.bed", help="Path to bed file.")
@click.option("--output_folder", type=str, default="/data/liminghong/sclab/sclab/random_test", help="Path to save dataset files.")
@click.option("--threads", type=int, default=8, help="Number of threads.")
def generate_domain_dataset(map_file, cools_path, hic_file, rna_file, boundary_file, output_folder, threads):
    random.seed(0)

    map_info = pd.read_csv(map_file, sep="\t", index_col=0)
    hic = anndata.read_h5ad(hic_file)
    rna = anndata.read_h5ad(rna_file)
    boundary = pd.read_csv(boundary_file, sep="\t", names=["chrom", "start", "end", "level"])

    with Pool(threads) as p:
        clrs = dict(p.map(get_cooler, [(hic_name, cools_path) for hic_name in hic.obs_names]))

    contacts = catch_locations(clrs, hic.obs_names, get_random_locations(boundary), threads)

    _dataset = []
    for scRNA, row in tqdm(map_info.iterrows(), total = map_info.shape[0]):
        _dataset.append({
            "scRNA": rna[scRNA, :].layers["counts"][0].astype(np.int32),
            "scRNA_head": rna.var_names,
            "scHiC": contacts.loc[row["scHiC"]].to_dict(),
            "cell_type": row["cell_type"],
            "identity": "truth",
        })

    hic_corr, rna_corr = get_corr_matrix(map_info, hic, rna, contacts)

    cell_type_count = map_info.groupby("cell_type").count()
    for cell_type, row in cell_type_count.iterrows():
        for i in range(min(cell_type_count.max()[0] - row["scHiC"], int(row["scHiC"]/2))):
            random_index = random.randint(0, row["scHiC"] - 1)
            _case = map_info[map_info["cell_type"]==cell_type].iloc[random_index]
            best_rna, best_hic = find_best_RNA(rna, rna_corr, _case.name, cell_type), find_best_HiC(contacts, hic_corr, _case["scHiC"], cell_type)
            
            _dataset.append({
                "scRNA": best_rna,
                "scRNA_head": rna.var_names,
                "scHiC": best_hic.loc[0].to_dict(),
                "cell_type": cell_type,
                "identity": "fake",
            })

    # Randomly generate train dataset and test dataset
    _indexs = random.sample(range(0, len(_dataset)), int(len(_dataset)*0.1))
    eval_dataset = [_dataset[_index] for _index in _indexs]
    _indexs = list(set(range(0, len(_dataset))) - set(_indexs))
    train_dataset = [_dataset[_index] for _index in _indexs]

    np.save(os.path.join(output_folder, "train_dataset.npy"), train_dataset)
    np.save(os.path.join(output_folder, "eval_dataset.npy"), eval_dataset)

@cli.command(short_help="Generate dataset.")
@click.option("--map_file", type=str, default="/data/liminghong/sclab/sclab/map_result.csv", help="Path to map_result.csv.")
@click.option("--cools_path", type=str, default="/data/liminghong/sclab/Lee2019/Human_single_cell_10kb_cool", help="Path to the folder containing cool files.")
@click.option("--hic_file", type=str, default="/data/liminghong/sclab/sclab/hic_result.h5ad", help="Path to hic_result.h5ad.")
@click.option("--rna_file", type=str, default="/data/liminghong/sclab/sclab/rna_result.h5ad", help="Path to rna_result.h5ad.")
@click.option("--annotation_file", type=str, default="/data/liminghong/sclab/public/gencode.v19.annotation.gtf", help="Path to gtf file.")
@click.option("--housekeeping_file", type=str, default="/data/liminghong/sclab/sclab/random_test/source/Housekeeping_GenesHuman.csv", help="Path to csv file.")
@click.option("--output_folder", type=str, default="/data/liminghong/sclab/sclab/random_test/hk", help="Path to save dataset files.")
@click.option("--threads", type=int, default=8, help="Number of threads.")
def generate_hk_dataset(map_file, cools_path, hic_file, rna_file, annotation_file, housekeeping_file, output_folder, threads):
    random.seed(0)

    map_info = pd.read_csv(map_file, sep="\t", index_col=0)
    hic = anndata.read_h5ad(hic_file)
    rna = anndata.read_h5ad(rna_file)
    v19_anno = pd.read_csv(
        annotation_file,
        header=None, sep='\t', skiprows=[i for i in range(5)], usecols=[0, 2, 3, 4, 6, 8],
        names=['chrom', 'type', 'start', 'end', 'strand', 'info']
    )
    v19_anno["gene_name"] = v19_anno["info"].str.extract(r'(gene_name ")(\w*)')[1]
    v19_anno = v19_anno[
        (v19_anno["type"]=="gene") &
        (v19_anno["gene_name"].isin(pd.read_csv(housekeeping_file, sep=";")["Gene.name"].values)) & 
        ((v19_anno["end"]-v19_anno["start"]) >= 200000)
    ]

    with Pool(threads) as p:
        clrs = dict(p.map(get_cooler, [(hic_name, cools_path) for hic_name in hic.obs_names]))

    locations = v19_anno.sample(n=12, random_state=0)[["chrom", "start", "end"]].values
    contacts = catch_locations(clrs, hic.obs_names, locations, threads)

    _dataset = []
    for scRNA, row in tqdm(map_info.iterrows(), total = map_info.shape[0]):
        _dataset.append({
            "scRNA": rna[scRNA, :].layers["counts"][0].astype(np.int32),
            "scRNA_head": rna.var_names,
            "scHiC": contacts.loc[row["scHiC"]].to_dict(),
            "cell_type": row["cell_type"],
            "identity": "truth",
        })

    hic_corr, rna_corr = get_corr_matrix(map_info, hic, rna, contacts)

    cell_type_count = map_info.groupby("cell_type").count()
    for cell_type, row in cell_type_count.iterrows():
        for i in range(min(cell_type_count.max()[0] - row["scHiC"], int(row["scHiC"]/2))):
            random_index = random.randint(0, row["scHiC"] - 1)
            _case = map_info[map_info["cell_type"]==cell_type].iloc[random_index]
            best_rna, best_hic = find_best_RNA(rna, rna_corr, _case.name, cell_type), find_best_HiC(contacts, hic_corr, _case["scHiC"], cell_type)
            
            _dataset.append({
                "scRNA": best_rna,
                "scRNA_head": rna.var_names,
                "scHiC": best_hic.loc[0].to_dict(),
                "cell_type": cell_type,
                "identity": "fake",
            })

    # Randomly generate train dataset and test dataset
    _indexs = random.sample(range(0, len(_dataset)), int(len(_dataset)*0.1))
    eval_dataset = [_dataset[_index] for _index in _indexs]
    _indexs = list(set(range(0, len(_dataset))) - set(_indexs))
    train_dataset = [_dataset[_index] for _index in _indexs]

    np.save(os.path.join(output_folder, "train_dataset.npy"), train_dataset)
    np.save(os.path.join(output_folder, "eval_dataset.npy"), eval_dataset)

@cli.command(short_help="Generate dataset of variable length.")
@click.option("--map_file", type=str, default="/data/liminghong/sclab/sclab/map_result.csv", help="Path to map_result.csv.")
@click.option("--cools_path", type=str, default="/data/liminghong/sclab/Lee2019/Human_single_cell_10kb_cool", help="Path to the folder containing cool files.")
@click.option("--hic_file", type=str, default="/data/liminghong/sclab/sclab/hic_result.h5ad", help="Path to hic_result.h5ad.")
@click.option("--rna_file", type=str, default="/data/liminghong/sclab/sclab/rna_result.h5ad", help="Path to rna_result.h5ad.")
@click.option("--output_folder", type=str, default="/data/liminghong/sclab/sclab/random_test/variable_length", help="Path to save dataset files.")
@click.option("--threads", type=int, default=8, help="Number of threads.")
def generate_variable_length_dataset(map_file, cools_path, hic_file, rna_file, output_folder, threads):
    random.seed(0)

    map_info = pd.read_csv(map_file, sep="\t", index_col=0)
    hic = anndata.read_h5ad(hic_file)
    rna = anndata.read_h5ad(rna_file)

    with Pool(threads) as p:
        clrs = dict(p.map(get_cooler, [(hic_name, cools_path) for hic_name in hic.obs_names]))
    
    bins = next(iter(clrs.values())).bins()[:]
    locations = []
    for length in range(50, 501, 50):
        _num = 0
        while _num < 5:
            index = random.sample(bins.index.tolist(), 1)[0]
            if bins.iloc[index].chrom != bins.iloc[index+length].chrom:
                continue
            locations.append((
                bins.iloc[index].chrom,
                bins.iloc[index].start,
                bins.iloc[index+length].start
            ))
            _num += 1

    contacts = catch_locations(clrs, hic.obs_names, locations, threads)

    _dataset = []
    for scRNA, row in tqdm(map_info.iterrows(), total = map_info.shape[0]):
        _dataset.append({
            "scRNA": rna[scRNA, :].layers["counts"][0].astype(np.int32),
            "scRNA_head": rna.var_names,
            "scHiC": contacts.loc[row["scHiC"]].to_dict(),
            "cell_type": row["cell_type"],
            "identity": "truth",
        })

    hic_corr, rna_corr = get_corr_matrix(map_info, hic, rna, contacts)

    cell_type_count = map_info.groupby("cell_type").count()
    for cell_type, row in cell_type_count.iterrows():
        for i in range(min(cell_type_count.max()[0] - row["scHiC"], int(row["scHiC"]/2))):
            random_index = random.randint(0, row["scHiC"] - 1)
            _case = map_info[map_info["cell_type"]==cell_type].iloc[random_index]
            best_rna, best_hic = find_best_RNA(rna, rna_corr, _case.name, cell_type), find_best_HiC(contacts, hic_corr, _case["scHiC"], cell_type)
            
            _dataset.append({
                "scRNA": best_rna,
                "scRNA_head": rna.var_names,
                "scHiC": best_hic.loc[0].to_dict(),
                "cell_type": cell_type,
                "identity": "fake",
            })

    # Randomly generate train dataset and test dataset
    _indexs = random.sample(range(0, len(_dataset)), int(len(_dataset)*0.1))
    eval_dataset = [_dataset[_index] for _index in _indexs]
    _indexs = list(set(range(0, len(_dataset))) - set(_indexs))
    train_dataset = [_dataset[_index] for _index in _indexs]

    np.save(os.path.join(output_folder, "train_dataset.npy"), train_dataset)
    np.save(os.path.join(output_folder, "eval_dataset.npy"), eval_dataset)


@cli.command(short_help="Generate dataset of variable length.")
@click.option("--map_file", type=str, default="/data/liminghong/sclab/sclab/map_result.csv", help="Path to map_result.csv.")
@click.option("--cools_path", type=str, default="/data/liminghong/sclab/Lee2019/Human_single_cell_10kb_cool", help="Path to the folder containing cool files.")
@click.option("--hic_file", type=str, default="/data/liminghong/sclab/sclab/hic_result.h5ad", help="Path to hic_result.h5ad.")
@click.option("--rna_file", type=str, default="/data/liminghong/sclab/sclab/rna_result.h5ad", help="Path to rna_result.h5ad.")
@click.option("--gene_name", type=str, default="PIP4K2A")
@click.option("--annotation_file", type=str, default="/data/liminghong/sclab/public/gencode.v19.annotation.gtf", help="Path to gtf file.")
@click.option("--output_folder", type=str, default="/data/liminghong/sclab/sclab/random_test/variable_length_around_gene", help="Path to save dataset files.")
@click.option("--threads", type=int, default=8, help="Number of threads.")
def generate_variable_length_around_gene_dataset(map_file, cools_path, hic_file, rna_file, gene_name, annotation_file, output_folder, threads):
    random.seed(0)

    map_info = pd.read_csv(map_file, sep="\t", index_col=0)
    hic = anndata.read_h5ad(hic_file)
    rna = anndata.read_h5ad(rna_file)
    v19_anno = pd.read_csv(
        annotation_file,
        header=None, sep='\t', skiprows=[i for i in range(5)], usecols=[0, 2, 3, 4, 6, 8],
        names=['chrom', 'type', 'start', 'end', 'strand', 'info']
    )
    v19_anno["gene_name"] = v19_anno["info"].str.extract(r'(gene_name ")(\w*)')[1]
    v19_anno = v19_anno[(v19_anno["type"]=="gene") & (v19_anno["gene_name"]==gene_name)]

    with Pool(threads) as p:
        clrs = dict(p.map(get_cooler, [(hic_name, cools_path) for hic_name in hic.obs_names]))
    
    bins = next(iter(clrs.values())).bins()[:]
    locations = []
    for length in range(0, 201, 40):
        _chrom, _start, _end = v19_anno["chrom"].item(), v19_anno["start"].item(), v19_anno["end"].item()
        _start -= length * 10000
        _end += length * 10000
        if _start < 0 or _end > bins[bins["chrom"]==_chrom]["start"].max():
            continue
        locations.append((_chrom, _start, _end))

    contacts = catch_locations(clrs, hic.obs_names, locations, threads)

    _dataset = []
    for scRNA, row in tqdm(map_info.iterrows(), total = map_info.shape[0]):
        _dataset.append({
            "scRNA": rna[scRNA, :].layers["counts"][0].astype(np.int32),
            "scRNA_head": rna.var_names,
            "scHiC": contacts.loc[row["scHiC"]].to_dict(),
            "cell_type": row["cell_type"],
            "identity": "truth",
        })

    # Randomly generate train dataset and test dataset
    _indexs = random.sample(range(0, len(_dataset)), int(len(_dataset)*0.1))
    eval_dataset = [_dataset[_index] for _index in _indexs]
    _indexs = list(set(range(0, len(_dataset))) - set(_indexs))
    train_dataset = [_dataset[_index] for _index in _indexs]

    np.save(os.path.join(output_folder, gene_name, "train_dataset.npy"), train_dataset)
    np.save(os.path.join(output_folder, gene_name, "eval_dataset.npy"), eval_dataset)

if __name__ == "__main__":
    cli()
