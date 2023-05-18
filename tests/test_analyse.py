import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scce.analyse import ig_attribute

from . import FileHelper


def test_ig_attribute():
    dataset = np.load(FileHelper().dataset_path, allow_pickle=True).item()["eval"]
    target_label = "PDGFRA"

    RNA_values = [data["scRNA"] for data in dataset if data["cell_type"] == "OPC"]
    gene_names = dataset[0]["scRNA_head"]
    ChRFs_score = ig_attribute(target_label, RNA_values, gene_names)
    ChRFs_score = ChRFs_score.sort_values(by="score", ascending=False)

    ax = sns.lineplot(data=ChRFs_score)
    ax.set_xticks([])
    ax.set_xticklabels([])
    plt.xlabel("genes")
    plt.ylabel("score")
    plt.savefig(FileHelper().ig_result_path, bbox_inches="tight")
