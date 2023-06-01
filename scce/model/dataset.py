from typing import Union

import numpy as np
import torch
import torch.utils.data as data

from scce.utils import LOGGER


def _crack(integer):
    start = int(np.sqrt(integer))
    factor = integer / start
    while int(factor) != factor:
        start += 1
        factor = integer / start
    return int(factor), start


class InputDataProcessingHelper:
    def __init__(self, input_length: int, kernel_size: int) -> None:
        self.input_length = input_length
        self.kernel_size = kernel_size

        _len = int(input_length / kernel_size / kernel_size)
        self.divided_input_length = _len * kernel_size * kernel_size
        self.patch_size = _crack(_len)
        self.divided_input_shape = tuple([i * kernel_size for i in self.patch_size])

    def do(self, input: np.array) -> np.array:
        return input[: self.divided_input_length].reshape(self.divided_input_shape)


class Dataset(data.Dataset):
    """
    Reading the training single-cell hic dataset
    """

    def __init__(
        self,
        datas_or_path: Union[str, list],
        target_label: str,
        kernel_size: int,
    ):
        super(Dataset, self).__init__()

        _datas = (
            datas_or_path
            if type(datas_or_path) is list
            else np.load(datas_or_path, allow_pickle=True)
        )
        self._get_data(_datas, target_label, kernel_size)

        LOGGER.info("Dataset is created.")

    def _get_data(self, datas, target_label, kernel_size):
        """
        [
            {
                'scRNA': np.array(),
                'scRNA_head': [],
                'scHiC':
                {
                    'target_label': np.array(),
                    ...
                }
            },
            ...
        ]
        """
        input_raw_length = datas[0]["scRNA"].shape[0]
        _helper = InputDataProcessingHelper(input_raw_length, kernel_size)

        self._scRNA_data, self._scHiC_data = [], []
        for _data in datas:
            _scHiC_data = _data["scHiC"][target_label]
            if np.all(_scHiC_data == 0):
                continue
            self._scHiC_data.append(_scHiC_data.tolist())

            self._scRNA_data.append(_helper.do(_data["scRNA"].copy()))

        self._scRNA_data = np.array(self._scRNA_data, dtype="float32")
        self._scHiC_data = np.array(self._scHiC_data, dtype="float32")

        self.input_raw_length = input_raw_length
        self.input_size = _helper.divided_input_shape
        self.patch_size = _helper.patch_size
        self.output_size = self._scHiC_data[0].shape[0]

    def __getitem__(self, index):
        input_tensor = torch.as_tensor(self._scRNA_data[index])
        target_tensor = torch.as_tensor(self._scHiC_data[index])
        return input_tensor, target_tensor

    def __len__(self):
        return len(self._scRNA_data)

    def slice(self, index):
        self._scRNA_data = self._scRNA_data[index]
        self._scHiC_data = self._scHiC_data[index]
