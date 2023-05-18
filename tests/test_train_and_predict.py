import anndata
import numpy as np
import pandas as pd
from scipy import stats

from scce import plot
from scce.model import DataSetGenerator, build, predict

from . import FileHelper


def test_generate_dataset():
    map = pd.read_csv(FileHelper().mapping_path, index_col=0)
    rna = anndata.read_h5ad(FileHelper().mapped_rna_path)
    hic_folder_path = "tests/data/hic"
    resolution = 10000
    target_label = "PDGFRA"

    generator = DataSetGenerator(map, rna, hic_folder_path, resolution)
    generator.generate_by_gene_name([target_label])
    dataset = generator.get_dataset()
    np.save(FileHelper().dataset_path, dataset)


def test_train_and_evaluate():
    dataset = np.load(FileHelper().dataset_path, allow_pickle=True).item()
    target_label = "PDGFRA"

    train_dataset, eval_dataset = dataset["train"], dataset["eval"]
    build(dict(train=train_dataset, eval=eval_dataset), target_label=target_label)
    evaluate = predict(dataset=eval_dataset, target_label=target_label)
    np.save(FileHelper().evaluate_predict_path, evaluate)


def test_plot():
    target_label = "PDGFRA"
    cell_types = ["Astro", "Endo", "OPC"]

    def cal_by_cell_type(targets, predicts, cell_type):
        _values = []
        for i in range(len(predicts)):
            pred = predicts[i]
            if targets[i]["cell_type"] != cell_type:
                continue
            _values.append(stats.pearsonr(pred, targets[i]["scHiC"][target_label])[0])
        return _values

    evaluate_target = np.load(FileHelper().dataset_path, allow_pickle=True).item()[
        "eval"
    ]
    evaluate_predict = np.load(FileHelper().evaluate_predict_path, allow_pickle=True)

    data = [
        cal_by_cell_type(evaluate_target, evaluate_predict, cell_type)
        for cell_type in cell_types
    ]
    plot.box(
        data,
        xticklabels=cell_types,
        output_path=FileHelper().predict_pearson_boxplot_path,
    )
