import os

import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display

from scce.plot.base import figure_size, set_Border, set_plt


def draw_pseudotime_line(values, xlabel=None, ylabel=None, save_dir_path=None):
    for i in range(len(values)):
        set_plt(figsize=(20, 10))
        sns.set_theme(style="whitegrid")

        fig, ax = plt.subplots()

        x = list(range(1, i + 2))
        if len(x) == 1:
            plt.scatter(x[0], values[: i + 1], linewidth=figure_size["small"])
        else:
            plt.plot(x, values[: i + 1], linewidth=figure_size["small"])

        set_Border(plt.gca())

        plt.xticks([i for i in range(1, len(values) + 1)])
        plt.xlim((0, len(values) + 1))
        plt.ylim((min(values) * 0.99, max(values) * 1.01))

        plt.tick_params(
            colors="black", bottom=True, left=True, labelsize=figure_size["ultra"]
        )
        plt.grid(False)

        if xlabel:
            plt.xlabel(xlabel, fontsize=figure_size["ultra"])
        if ylabel:
            plt.ylabel(ylabel, fontsize=figure_size["ultra"])

        if save_dir_path:
            plt.savefig(
                "{}.pdf".format(os.path.join(save_dir_path, str(i + 1))),
                bbox_inches="tight",
            )

        display.clear_output(wait=True)
        plt.pause(0.00000001)
