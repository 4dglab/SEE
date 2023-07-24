import os
from typing import List

import numpy as np
from scipy import interpolate

from scce.utils import array2mat

from .attribution import integrated_gradients
from .chromatin_remodeling import get_primary_interactions, get_value_matrix


def ig_attribute(target_label: str, RNA_values: List[np.array], gene_names: List[str]):
    return integrated_gradients(
        os.path.join(".scce", target_label, "model.pth"), RNA_values, gene_names
    )


def interaction_diffusion(
    hics: np.array,
    pseudotimes: np.array,
    cell_length: float = 0.5,
    random_seed: int = 42,
):
    primary_interactions = get_primary_interactions(hics, pseudotimes, random_seed)

    start, end = 0, array2mat(hics[0]).shape[0]
    xx, yy, value = get_value_matrix(primary_interactions, start, end, cell_length)

    xx, yy, value = xx.ravel(), yy.ravel(), value.ravel()
    xx, yy, value = (
        list(xx[~np.isnan(xx)]),
        list(yy[~np.isnan(yy)]),
        list(value[~np.isnan(value)]),
    )

    x = y = np.linspace(start, end, int((end - start) / cell_length * 5 + 1))
    value = interpolate.griddata(
        (xx, yy), value, (x[None, :], y[:, None]), method="cubic"
    )

    return x, y, value
