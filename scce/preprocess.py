import anndata
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from scce.data import HiCLoader


def load_hics(folder_path, file_names, resolution, n_jobs=1):
    def _load_hic(folder_path, file_name):
        _hic_loader = HiCLoader(folder_path, resolution)
        _c = _hic_loader.load_hic(file_name)
        binsize, chromosome_lengths = _c.binsize, _c.chromosome_lengths
        contact = _hic_loader.get_contact(_c)

        if binsize != resolution:
            raise ValueError(f"File {file_name} resolution is not {resolution}.")

        return _get_contact(contact, binsize, chromosome_lengths).rename(
            columns={0: file_name}
        )

    def _get_contact(contact, binsize, chromosome_lengths):
        contact = contact[contact["start1"] != contact["start2"]]
        contact = contact[contact["chrom1"] == contact["chrom2"]]

        _1 = contact.groupby(["chrom1", "start1"])["count"].sum()
        _2 = contact.groupby(["chrom2", "start2"])["count"].sum()
        _1.index.names = _2.index.names = ["chrom", "start"]
        _1, _2 = _1[_1 != 0], _2[_2 != 0]
        info = pd.concat([_1, _2], axis=1).fillna(0).sum(axis=1).sort_index()

        _indexs = set(
            [
                (chrom, int(i * binsize))
                for chrom in chromosome_lengths.keys()
                for i in range(int(chromosome_lengths[chrom] / binsize) + 1)
            ]
        )
        _indexs = pd.MultiIndex.from_tuples(_indexs - set(info.index))
        mutual_info = pd.DataFrame(np.zeros(len(_indexs)), index=_indexs)
        mutual_info.index.rename(["chrom", "start"], inplace=True)
        info = pd.concat([info, mutual_info]).sort_index()

        return info.astype("float16")

    joblist = [delayed(_load_hic)(folder_path, file_name) for file_name in file_names]

    infos = Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(joblist)
    infos = pd.concat(infos, axis=1).fillna(0).sort_index()
    return infos


def hic_process(
    metadata_path,
    hic_folder_path,
    output_path,
    column_names=dict(id="sample_name", cell_type="cell_type"),
    cell_types=[],
    resolution=10000,
    n_jobs=1,
):
    sample_name, cell_type = column_names["id"], column_names["cell_type"]

    _metadata = pd.read_csv(metadata_path)
    if cell_types:
        _metadata = _metadata[_metadata[cell_type].isin(cell_types)]
    if _metadata.empty:
        raise ValueError("No valid sample in metadata.")

    infos = load_hics(hic_folder_path, _metadata[sample_name], resolution, n_jobs)

    obs = _metadata.set_index(sample_name)
    obs = obs.loc[infos.T.index]
    obs["cell_type"], obs["domain"] = obs[cell_type], "scHiC"
    var = infos.reset_index()[["chrom", "start"]].set_index(
        infos.index.map("{0[0]}_{0[1]}".format)
    )
    var["end"] = var["start"] + resolution

    infos.index = infos.index.map("{0[0]}_{0[1]}".format)
    infos = anndata.AnnData(X=infos.T, obs=obs, var=var, dtype="float16")

    infos.write(output_path, compression="gzip")


def rna_process(
    metadata_path,
    matrix_path,
    output_path,
    column_names=dict(id="sample_name", cell_type="cell_type"),
    cell_types=[],
):
    sample_name, cell_type = column_names["id"], column_names["cell_type"]

    _metadata = pd.read_csv(metadata_path)
    if cell_types:
        _metadata = _metadata[_metadata[cell_type].isin(cell_types)]

    infos = pd.DataFrame()
    for chunk in pd.read_csv(matrix_path, chunksize=10000):
        _filter = chunk[chunk[sample_name].isin(_metadata[sample_name])]
        infos = pd.concat([infos, _filter])
    infos = infos.set_index(sample_name)

    obs = _metadata.set_index(sample_name)
    obs = obs.loc[infos.index]
    obs["cell_type"], obs["domain"] = obs[cell_type], "scRNA"

    infos = anndata.AnnData(X=infos, obs=obs, dtype="int32")

    infos.write(output_path, compression="gzip")
