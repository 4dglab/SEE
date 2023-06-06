import os
from typing import List

import numpy as np

from .attribution import integrated_gradients


def ig_attribute(target_label: str, RNA_values: List[np.array], gene_names: List[str]):
    return integrated_gradients(
        os.path.join(".scce", target_label, "model.pth"), RNA_values, gene_names
    )
