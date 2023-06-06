from typing import List

import numpy as np
import torch

from scce.model.dataset import InputDataProcessingHelper
from scce.model.net import load_network


def evaluate(model_file, RNA_values: List[np.array]):
    checkpoint, model = load_network(model_file)
    model.cuda()
    model.eval()

    input_raw_length, kernel_size = (
        checkpoint["input_raw_length"],
        checkpoint["kernel_size"],
    )
    helper = InputDataProcessingHelper(input_raw_length, kernel_size)

    output_data = []
    for RNA_value in RNA_values:
        input = helper.do(RNA_value.copy())
        input = torch.Tensor(input).cuda().unsqueeze(0).unsqueeze(0)

        output = model(input).detach().cpu().numpy()
        output_data.append(output[0, 0])

    output_data = np.array(output_data)
    return output_data
