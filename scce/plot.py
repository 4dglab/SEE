import matplotlib.pyplot as plt
import scanpy as sc


def umap(anndata: sc.AnnData, umap_kwargs: dict, output_path: str = None):
    anndata = anndata.copy()
    sc.pp.neighbors(anndata, use_rep="X_scce", metric="cosine")
    sc.tl.umap(anndata)
    sc.pl.umap(anndata, **umap_kwargs)
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
