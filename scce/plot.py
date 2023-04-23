import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

figure_size = dict(
    ultra=45,
    large=35,
    middle=20,
    small=10,
    very_small=5
)

def set_plt(figsize=(10, 10)):
    params = {'axes.titlesize': figure_size['ultra'],
              'legend.fontsize': figure_size['middle'],
              'figure.figsize': figsize,
              'axes.labelsize': figure_size['ultra'],
              'xtick.labelsize': figure_size['ultra'],
              'ytick.labelsize': figure_size['ultra'],
              'figure.titlesize': figure_size['ultra'],
              'lines.linewidth': figure_size['very_small']}
    plt.rcParams.update(params)

def set_Border(axes):
    axes.spines['top'].set_color('none')
    axes.spines['right'].set_color('none')
    axes.spines['bottom'].set_color('black')
    axes.spines['left'].set_color('black')
    axes.spines['bottom'].set_linewidth(figure_size['very_small'])
    axes.spines['left'].set_linewidth(figure_size['very_small'])
    axes.tick_params(axis='both', width=figure_size['very_small'], length=figure_size['small'])

def draw_pseudotime_line(values, xlabel=None, ylabel=None, save_dir_path=None):
    import os
    from IPython import display

    for i in range(len(values)):
        set_plt(figsize=(20, 10))
        sns.set_theme(style="whitegrid")

        fig, ax = plt.subplots()

        x = list(range(1, i+2))
        if len(x) == 1:
            plt.scatter(x[0], values[:i+1], linewidth=figure_size['small'])
        else:
            plt.plot(x, values[:i+1], linewidth=figure_size['small'])

        set_Border(plt.gca())

        plt.xticks([i for i in range(1, len(values)+1)])
        plt.xlim((0, len(values)+1))
        plt.ylim((min(values)*0.99, max(values)*1.01))
        
        plt.tick_params(colors='black', bottom=True, left=True, labelsize=figure_size['ultra'])
        plt.grid(False)

        if xlabel:
            plt.xlabel(xlabel, fontsize=figure_size['ultra'])
        if ylabel:
            plt.ylabel(ylabel, fontsize=figure_size['ultra'])

        if save_dir_path:
            plt.savefig('{}.pdf'.format(os.path.join(save_dir_path, str(i+1))), bbox_inches='tight')
        
        display.clear_output(wait=True)
        plt.pause(0.00000001)


def umap(anndata: sc.AnnData, umap_kwargs: dict, output_path: str = None):
    anndata = anndata.copy()
    sc.pp.neighbors(anndata, use_rep="X_scce", metric="cosine")
    sc.tl.umap(anndata)
    sc.pl.umap(anndata, **umap_kwargs)
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')


def box(anndata: sc.AnnData, x, y, hue, output_path: str = None):
    set_plt(figsize=(10, 10))
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots()

    ax = sns.boxplot(data=anndata, x=x, y=y, hue=hue, linewidth=figure_size['very_small'])

    set_Border(plt.gca())
    plt.tick_params(colors='black', bottom=True, left=True, labelsize=figure_size['ultra'])
    plt.legend(
        frameon=False, markerscale=2, borderpad=1, borderaxespad=0, fontsize=figure_size['middle'], loc='lower right')
    plt.grid(False)

    plt.xlabel(x, fontsize=figure_size['ultra'])
    plt.ylabel(y, fontsize=figure_size['ultra'])

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()
