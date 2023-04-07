import logging
from typing import Dict, Union

import numpy as np
import torch
import torch.utils.data as data


class Dataset(data.Dataset):
    '''
    Reading the training single-cell hic dataset
    '''
    def __init__(self, datas_or_path: Union[Dict, str], target_label: str, is_train: bool = False):
        super(Dataset, self).__init__()

        self.is_train = is_train
        _datas = datas_or_path if type(datas_or_path) is Dict else np.load(datas_or_path, allow_pickle=True)
        self._get_data(_datas, target_label)
        
        logger = logging.getLogger('base')
        logger.info('Dataset is created.')

    def _get_data(self, datas, target_label):
        '''
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
        '''
        def _crack(integer):
            start = int(np.sqrt(integer))
            factor = integer / start
            while int(factor) != factor:
                start += 1
                factor = integer / start
            return int(factor), start

        self._scRNA_data, self._scHiC_data = [], []
        for _data in datas:
            if self.is_train:
                _scHiC_data = _data['scHiC'][target_label]
                if np.all(_scHiC_data == 0):
                    continue
                self._scHiC_data.append(_scHiC_data.tolist())

            _scRNA, _scRNA_head = _data['scRNA'].copy(), _data['scRNA_head']
            # _where = np.where(_scRNA_head.isin(list( _filter_genes.tolist())))
            # _where = np.array(list(set(list(range(_scRNA.shape[0]))) - set(_where[0])))
            # _scRNA[_where] = 0
            _len = int(_scRNA.shape[0] / 64)
            _input_size = tuple([i * 8 for i in _crack(_len)])
            self._scRNA_data.append(np.array(_scRNA[:_len*64].reshape(_input_size)))

        self._scRNA_data = np.array(self._scRNA_data, dtype='float32')
        self._scHiC_data = np.array(self._scHiC_data, dtype='float32')

    def __getitem__(self, index):
        input_tensor = torch.as_tensor(self._scRNA_data[index])
        target_tensor = torch.as_tensor(self._scHiC_data[index]) if self.is_train else torch.as_tensor([])
        return input_tensor, target_tensor

    def __len__(self):
        return len(self._scRNA_data)

    def slice(self, index):
        self._scRNA_data = self._scRNA_data[index]
        self._scHiC_data = self._scHiC_data[index]

