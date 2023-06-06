import collections
import itertools
from typing import Any, Callable, Mapping, Optional, Union

import anndata
import networkx as nx
import numpy as np
import pandas as pd
import pybedtools
import scanpy as sc
import scglue
from pybedtools.cbedtools import Interval
from scglue.genomics import Bed
from tqdm.auto import tqdm


class DataTool:
    def __init__(self, hic_process_path, rna_process_path, cell_types):
        hic, rna = anndata.read_h5ad(hic_process_path), anndata.read_h5ad(
            rna_process_path
        )
        if cell_types:
            hic = hic[hic.obs["cell_type"].isin(cell_types), :]
            rna = rna[rna.obs["cell_type"].isin(cell_types), :]
        if "counts" not in rna.layers:
            rna.layers["counts"] = rna.X.copy()

        self.hic, self.rna = hic, rna

    def add_hic_pca(self, hic_pca_path):
        hic_pca = pd.read_csv(hic_pca_path, index_col=0)
        hic_pca = hic_pca.loc[self.hic.obs.index]
        self.hic.obsm["X_pca"] = hic_pca.to_numpy()

    def add_gene_annotation(self, gtf_path, gtf_by="gene_symbol"):
        scglue.data.get_gene_annotation(self.rna, gtf=gtf_path, gtf_by=gtf_by)
        self.rna = self.rna[:, self.rna.var.dropna(axis=0, how="all").index]

    def hic_pca(self, metric="cosine"):
        sc.pp.neighbors(self.hic, use_rep="X_pca", metric=metric)
        sc.tl.umap(self.hic)
        sc.tl.leiden(self.hic)

    def rna_pca(self, metric="cosine"):
        self.rna.X = self.rna.layers["counts"].copy()
        sc.pp.highly_variable_genes(self.rna, n_top_genes=2000, flavor="seurat_v3")
        sc.pp.normalize_total(self.rna)
        sc.pp.log1p(self.rna)
        sc.pp.scale(self.rna)
        sc.tl.pca(self.rna, n_comps=100, svd_solver="auto")

        sc.pp.neighbors(self.rna, metric=metric)
        sc.tl.umap(self.rna)
        sc.tl.leiden(self.rna)

    def rna_highly_variable_genes(self):
        sc.tl.rank_genes_groups(self.rna, "cell_type", method="t-test")
        marker_genes = pd.DataFrame(self.rna.uns["rank_genes_groups"]["names"])[:1000]
        marker_genes_index = marker_genes.values.reshape(-1)
        _index = set(self.rna.var[self.rna.var["highly_variable"] == True].index) & set(
            marker_genes_index
        )
        self.rna.var["highly_variable"] = False
        self.rna.var.loc[list(_index), "highly_variable"] = True

    def get_data(self):
        return self.hic, self.rna


def interval_dist(x: Interval, y: Interval, resolution: int) -> int:
    if x.chrom != y.chrom:
        return np.inf * (-1 if x.chrom < y.chrom else 1)

    _start1, _ = (x.stop, x.start) if x.strand == "-" else (x.start, x.stop)
    _start2, _ = (y.stop, y.start) if y.strand == "-" else (y.start, y.stop)

    x_range = set(
        [
            i
            for i in range(
                int(x.start / resolution) * resolution,
                int(x.end / resolution) * resolution + resolution,
                resolution,
            )
        ]
    )
    y_range = set([i for i in range(y.start, y.end + resolution, resolution)])
    if x_range & y_range:
        return 0

    return _start1 - _start2


def window_graph(
    left: Union[Bed, str],
    right: Union[Bed, str],
    window_size: int,
    resolution: int,
    left_sorted: bool = False,
    right_sorted: bool = False,
    attr_fn: Optional[Callable[[Interval, Interval, float], Mapping[str, Any]]] = None,
) -> nx.MultiDiGraph:
    if isinstance(left, Bed):
        pbar_total = len(left)
        left = left.to_bedtool()
    else:
        pbar_total = None
        left = pybedtools.BedTool(left)
    if not left_sorted:
        left = left.sort(stream=True)
    left = iter(left)  # Resumable iterator
    if isinstance(right, Bed):
        right = right.to_bedtool()
    else:
        right = pybedtools.BedTool(right)
    if not right_sorted:
        right = right.sort(stream=True)
    right = iter(right)  # Resumable iterator

    attr_fn = attr_fn or (lambda l, r, d: {})
    if pbar_total is not None:
        left = tqdm(left, total=pbar_total, desc="window_graph")
    graph = nx.MultiDiGraph()
    window = collections.OrderedDict()  # Used as ordered set
    searched_chrom = set()
    for l in left:
        searched_chrom.add(l.chrom)
        for r in list(window.keys()):  # Allow remove during iteration
            if r.chrom != l.chrom and r.chrom in searched_chrom:
                del window[r]
                continue
            d = interval_dist(l, r, resolution)
            if -window_size <= d <= window_size:
                graph.add_edge(l.name, r.name, **attr_fn(l, r, d))
        else:
            for r in right:  # Resume from last break
                d = interval_dist(l, r, resolution)
                window[r] = None
                if np.isinf(d):
                    break
                if -window_size <= d <= window_size:
                    graph.add_edge(l.name, r.name, **attr_fn(l, r, d))
    pybedtools.cleanup()
    return graph


def generate_graph(hic: anndata.AnnData, rna: anndata.AnnData, resolution: int = 10000):
    rna_bed = scglue.genomics.Bed(rna.var.assign(name=rna.var_names))
    hic_bed = scglue.genomics.Bed(hic.var.assign(name=hic.var_names))

    rna_bed = rna_bed.expand(5000, 0)

    def _dist_power_decay(x: int) -> float:
        return ((x + resolution) / resolution) ** (-0.75)

    graph = window_graph(
        rna_bed,
        hic_bed,
        window_size=resolution,
        resolution=resolution,
        attr_fn=lambda l, r, d, s=1: {
            "dist": abs(d),
            "weight": _dist_power_decay(abs(d)),
            "sign": s,
        },
    )

    rgraph = graph.reverse()
    nx.set_edge_attributes(graph, "fwd", name="type")
    nx.set_edge_attributes(rgraph, "rev", name="type")
    graph = scglue.graph.compose_multigraph(graph, rgraph)

    hvg_reachable = scglue.graph.reachable_vertices(
        graph, rna.var.query("highly_variable").index
    )
    hic.var["highly_variable"] = [item in hvg_reachable for item in hic.var_names]

    all_features = set(
        itertools.chain.from_iterable(map(lambda x: x.var_names, [rna, hic]))
    )
    for item in all_features:
        graph.add_edge(item, item, weight=1.0, sign=1, type="loop")

    return graph


def glue_embedding(
    hic: anndata.AnnData,
    rna: anndata.AnnData,
    graph: nx.MultiDiGraph,
    CPU_ONLY: bool = True,
):
    scglue.config.CPU_ONLY = CPU_ONLY
    scglue.models.configure_dataset(
        rna, "NB", use_highly_variable=True, use_layer="counts", use_rep="X_pca"
    )
    scglue.models.configure_dataset(
        hic, "NB", use_highly_variable=True, use_rep="X_pca"
    )
    graph = graph.subgraph(
        itertools.chain(
            rna.var.query("highly_variable").index,
            hic.var.query("highly_variable").index,
        )
    )
    glue = scglue.models.fit_SCGLUE(
        {"rna": rna, "hic": hic},
        graph,
        fit_kws={
            "directory": "glue",
        },
    )

    rna.obsm["X_scce"] = glue.encode_data("rna", rna)
    hic.obsm["X_scce"] = glue.encode_data("hic", hic)


def mapping(hic: anndata.AnnData, rna: anndata.AnnData) -> pd.DataFrame:
    combined = anndata.concat([rna, hic])

    sc.pp.neighbors(combined, use_rep="X_scce", metric="cosine")
    sc.tl.umap(combined)

    _map = pd.DataFrame()
    for cell_type in combined.obs.cell_type.unique():
        _sub = combined[
            combined.obs[combined.obs["cell_type"] == cell_type].index, :
        ].copy()
        _con = pd.DataFrame(
            _sub.obsp["connectivities"].toarray(),
            columns=_sub.obs_names,
            index=_sub.obs_names,
        )
        _con = _con.loc[
            _sub.obs[_sub.obs["domain"] == "scRNA"].index,
            _sub.obs[_sub.obs["domain"] == "scHiC"].index,
        ]
        _con = _con[_con.apply(np.sum, axis=1) != 0]
        if _con.index.empty:
            continue
        _result = pd.DataFrame(_con.idxmax(1), columns=["scHiC"])
        _result["cell_type"] = cell_type
        _map = pd.concat([_map, _result])

    return _map
