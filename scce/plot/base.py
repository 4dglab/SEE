import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
figure_size = dict(
    ultra=45,
    large=35,
    middle=20,
    small=10,
    very_small=5,
)


def set_plt(figsize=(10, 10)):
    params = {
        "axes.titlesize": figure_size["ultra"],
        "legend.fontsize": figure_size["middle"],
        "figure.figsize": figsize,
        "axes.labelsize": figure_size["ultra"],
        "xtick.labelsize": figure_size["ultra"],
        "ytick.labelsize": figure_size["ultra"],
        "figure.titlesize": figure_size["ultra"],
        "lines.linewidth": figure_size["very_small"],
    }
    plt.rcParams.update(params)


def set_Border(axes):
    axes.spines["top"].set_color("none")
    axes.spines["right"].set_color("none")
    axes.spines["bottom"].set_color("black")
    axes.spines["left"].set_color("black")
    axes.spines["bottom"].set_linewidth(figure_size["very_small"])
    axes.spines["left"].set_linewidth(figure_size["very_small"])
    axes.tick_params(
        axis="both", width=figure_size["very_small"], length=figure_size["small"]
    )


def umap(anndata: sc.AnnData, umap_kwargs: dict, output_path: str = None):
    anndata = anndata.copy()
    sc.pp.neighbors(anndata, use_rep="X_scce", metric="cosine")
    sc.tl.umap(anndata)
    sc.pl.umap(anndata, **umap_kwargs)
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")


def box(data, x=None, y=None, hue=None, xticklabels=None, output_path: str = None):
    set_plt(figsize=(10, 10))
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots()

    ax = sns.boxplot(data=data, x=x, y=y, hue=hue, linewidth=figure_size["very_small"])

    set_Border(plt.gca())
    plt.tick_params(
        colors="black", bottom=True, left=True, labelsize=figure_size["ultra"]
    )
    plt.legend(
        frameon=False,
        markerscale=2,
        borderpad=1,
        borderaxespad=0,
        fontsize=figure_size["middle"],
        loc="lower right",
    )
    plt.grid(False)

    if x:
        plt.xlabel(x, fontsize=figure_size["ultra"])
    if y:
        plt.ylabel(y, fontsize=figure_size["ultra"])
    if xticklabels:
        ax.set_xticklabels(labels=xticklabels)

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    else:
        plt.show()


def heatmap(x, y, value, levels, cbar_label, output_path: str = None):
    set_plt(figsize=(10, 10))
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots()

    cs = ax.contourf(x, y, value, levels=levels, cmap=plt.cm.jet)
    cbar = plt.colorbar(cs)
    cbar.set_label(cbar_label, rotation=90, fontsize=15)

    ax.xaxis.set_ticks_position("top")
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    plt.tick_params(
        colors="black",
        top=True,
        bottom=False,
        left=True,
        labelsize=figure_size["small"],
    )

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    else:
        plt.show()
