import os
import random
from typing import Dict, List, Tuple

import anndata
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from scce.data import HiCLoader
from scce.model.train_model import train
from scce.model.validate import evaluate
from scce.utils import mat2array


class DataSetGenerator:
    def __init__(
        self,
        map: pd.DataFrame,
        rna: anndata.AnnData,
        hic_folder_path: str,
        resolution: int = 10000,
    ):
        self.map = map
        self.rna = rna
        self.hic_loader = HiCLoader(hic_folder_path, resolution)
        self.resolution = resolution

    def _get_hics(self, hic_names: List[str], n_jobs: int) -> Dict:
        def _get_hic(file_name: str):
            return file_name, self.hic_loader.load_hic(file_name)

        joblist = [delayed(_get_hic)(hic_name) for hic_name in hic_names]
        infos = Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(joblist)
        return dict(infos)

    def _catch_locations(
        self, hic_names: List[str], locations: Dict, n_jobs: int
    ) -> pd.DataFrame:
        def _catch_location(
            hic_name, chrom, chromStart, chromEnd, label
        ) -> Tuple[str, np.ndarray]:
            mat = self.hic_loader.catch_matrix(
                self.hics[hic_name], chrom, chromStart, chromEnd
            )
            return label, mat2array(mat).astype(np.uint16)

        _contacts = pd.DataFrame()
        for hic_name in tqdm(hic_names):
            joblist = [
                delayed(_catch_location)(
                    hic_name,
                    location["chrom"],
                    location["chromStart"],
                    location["chromEnd"],
                    location["label"],
                )
                for location in locations
            ]
            infos = Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(joblist)
            _contact = pd.DataFrame([dict(infos)], index=[hic_name])
            _contacts = pd.concat([_contacts, _contact], axis=0)
        return _contacts

    def _generate_dataset(self, contacts: pd.DataFrame, eval_preportion: float):
        _dataset = []
        for scRNA, row in tqdm(self.map.iterrows(), total=self.map.shape[0]):
            _dataset.append(
                {
                    "scRNA": self.rna[scRNA, :].layers["counts"][0].astype(np.int32),
                    "scRNA_head": self.rna.var_names,
                    "scHiC": contacts.loc[row["scHiC"]].to_dict(),
                    "cell_type": row["cell_type"],
                }
            )

        # Randomly generate train dataset and test dataset
        _indexs = random.sample(
            range(0, len(_dataset)), int(len(_dataset) * eval_preportion)
        )
        self.eval_dataset = [_dataset[_index] for _index in _indexs]
        _indexs = list(set(range(0, len(_dataset))) - set(_indexs))
        self.train_dataset = [_dataset[_index] for _index in _indexs]

    def generate_by_gene_name(
        self, gene_names: List[str], n_jobs: int = 1, eval_preportion: float = 0.1
    ):
        _hic_names = self.map["scHiC"].unique().tolist()
        self.hics = self._get_hics(_hic_names, n_jobs)
        locations = [
            {
                "chrom": self.rna.var.loc[gene_name]["chrom"],
                "chromStart": int(self.rna.var.loc[gene_name]["chromStart"]),
                "chromEnd": int(self.rna.var.loc[gene_name]["chromEnd"]),
                "label": gene_name,
            }
            for gene_name in gene_names
        ]
        contacts = self._catch_locations(_hic_names, locations, n_jobs)

        self._generate_dataset(contacts, eval_preportion)

    def generate_by_location(
        self,
        locations: List[Tuple[str, int, int]],
        n_jobs: int = 1,
        eval_preportion: float = 0.1,
    ):
        _hic_names = self.map["scHiC"].unique().tolist()
        self.hics = self._get_hics(_hic_names, n_jobs)
        locations = [
            {
                "chrom": location[0],
                "chromStart": location[1],
                "chromEnd": location[2],
                "label": "{}_{}_{}".format(location[0], location[1], location[2]),
            }
            for location in locations
        ]
        contacts = self._catch_locations(_hic_names, locations, n_jobs)

        self._generate_dataset(contacts, eval_preportion)

    def get_dataset(self):
        return dict(train=self.train_dataset, eval=self.eval_dataset)


def build(dataset: Dict, target_label: str):
    train(
        dataset["train"],
        dataset["eval"],
        os.path.join(".scce", target_label),
        target_label,
    )


def predict(target_label: str, RNA_values: List[np.array]):
    return evaluate(os.path.join(".scce", target_label, "model.pth"), RNA_values)
