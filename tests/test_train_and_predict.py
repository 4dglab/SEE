import anndata
import numpy as np
import pandas as pd

from scce import plot
from scce.model import build, DataSetGenerator, predict
from . import FileHelper


def test_generate_and_train_and_evaluate():
    map = pd.read_csv(FileHelper().mapping_path, index_col=0)
    rna = anndata.read_h5ad(FileHelper().mapped_rna_path)
    hic_folder_path = 'tests/data/hic'
    resolution = 10000

    generator = DataSetGenerator(map, rna, hic_folder_path, resolution)
    generator.generate_by_gene_name(['PDGFRA'])
    train_dataset, eval_dataset = generator.get_dataset()
    build(dict(train=train_dataset, eval=eval_dataset), target_label='PDGFRA')
    evaluate = predict(dataset=eval_dataset, target_label='PDGFRA')
    np.save(FileHelper().evaluate_path, evaluate)


def test_plot():
    evaluate = np.load(FileHelper().evaluate_path)
    plot.box(evaluate, )