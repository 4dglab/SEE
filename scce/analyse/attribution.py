import math
from typing import List

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients

from scce.model.dataset import InputDataProcessingHelper
from scce.model.net import load_network


def integrated_gradients(
    model_file, RNA_values: List[np.array], gene_names: List[str]
) -> pd.DataFrame:
    checkpoint, model = load_network(model_file)
    model.cuda()
    model.eval()
    ig = IntegratedGradients(model)

    input_raw_length, kernel_size, input_size, output_size = (
        checkpoint["input_raw_length"],
        checkpoint["kernel_size"],
        checkpoint["input_size"],
        checkpoint["output_size"],
    )
    helper = InputDataProcessingHelper(input_raw_length, kernel_size)

    scores = torch.zeros(math.prod(input_size))
    for RNA_value in RNA_values:
        input = helper.do(RNA_value.copy())
        input = torch.Tensor(input).cuda().unsqueeze(0).unsqueeze(0)

        for i in range(output_size):
            attributions = ig.attribute(input, target=(0, i))[0, 0].reshape(-1)
            scores += attributions.cpu().detach()

    scores = pd.DataFrame(scores, index=gene_names[: len(scores)], columns=["score"])
    return scores
