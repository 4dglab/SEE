import anndata
import numpy as np
import pandas as pd

from scce import plot
from scce.model import build, DataSetGenerator, predict
from . import FileHelper


def test_generate_dataset():
    map = pd.read_csv(FileHelper().mapping_path, index_col=0)
    rna = anndata.read_h5ad(FileHelper().mapped_rna_path)
    hic_folder_path = 'tests/data/hic'
    resolution = 10000
    target_label = 'PDGFRA'

    generator = DataSetGenerator(map, rna, hic_folder_path, resolution)
    generator.generate_by_gene_name([target_label])
    dataset = generator.get_dataset()
    np.save(FileHelper().dataset_path, dataset)


def test_train_and_evaluate():
    dataset = np.load(FileHelper().dataset_path, allow_pickle=True).item()
    target_label = 'PDGFRA'

    train_dataset, eval_dataset = dataset['train'], dataset['eval']
    build(dict(train=train_dataset, eval=eval_dataset), target_label=target_label)
    evaluate = predict(dataset=eval_dataset, target_label=target_label)
    np.save(FileHelper().evaluate_path, evaluate)


def test_plot():
    evaluate = np.load(FileHelper().evaluate_path)
    plot.box(evaluate, )
