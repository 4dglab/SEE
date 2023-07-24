import random

import numpy as np
import pandas as pd

from scce.utils import array2mat


def get_primary_interactions(
    hics: np.array, pseudotimes: np.array, random_seed: int = 42
) -> pd.DataFrame:
    assert (
        hics.shape[0] == pseudotimes.shape[0]
    ), "Error with input length {} vs {}".format(hics.shape[0], pseudotimes.shape[0])

    random.seed(random_seed)
    np.random.seed(random_seed)

    # sort
    sort_indices = np.argsort(pseudotimes)
    _hics, _pseudotimes = hics[sort_indices], pseudotimes[sort_indices]

    result = pd.DataFrame()
    for sample_index in range(hics.shape[0]):
        _hic = array2mat(_hics[sample_index]).copy()
        _pseudotime = _pseudotimes[sample_index]

        if _hic.min() < 0:
            _hic += abs(_hic.min())
        else:
            _hic -= abs(_hic.min())

        _where = np.where(np.ones(_hic.shape))
        _where = list(zip(_where[0], _where[1]))
        _test = np.zeros(len(_where), dtype=object)
        for i in range(len(_where)):
            _test[i] = _where[i]
        _where = _test
        _p = np.triu(_hic, 1).reshape(-1)
        _p /= _p.sum()

        _result = []
        for i, j in np.random.choice(_where, max(1, int(_hic.shape[0] / 20)), p=_p):
            _i, _j = random.random() + i, random.random() + j
            if i == j:
                _i, _j = max(_i, _j), min(_i, _j)
            _result.append([sample_index, _i, _j, _pseudotime])
            _result.append([sample_index, _j, _i, _pseudotime])
        result = pd.concat(
            [
                result,
                pd.DataFrame(_result, columns=["sample_index", "i", "j", "pseudotime"]),
            ]
        )
    return result


def get_value_matrix(
    primary_interactions: pd.DataFrame,
    start: int,
    end: int,
    cell_length: float = 0.5,
):
    def _norm(number, resolution):
        return round(number / resolution) * resolution

    x = y = np.linspace(start, end, int((end - start) / cell_length + 1))
    yy, xx = np.meshgrid(x, y)

    count, value = np.zeros(xx.shape), np.zeros(xx.shape)
    for _, row in primary_interactions.iterrows():
        _x = int(_norm(row["i"], cell_length) / cell_length)
        _y = int(_norm(row["j"], cell_length) / cell_length)
        value[_x, _y] += row["pseudotime"]
        count[_x, _y] += 1
    for i in range(value.shape[0]):
        for j in range(value.shape[1]):
            if count[i, j] != 0:
                value[i, j] /= count[i, j]

    xx[count == 0] = yy[count == 0] = value[count == 0] = np.nan
    return xx, yy, value
