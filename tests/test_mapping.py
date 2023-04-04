import pytest
from scce.mapping import preprocess


def test_hic_process():
    hic_folder_path = 'tests/data/hic'
    output_path = 'tests/data/hic.h5'
    with pytest.raises(ValueError, match=r".* resolution is not .*"):
        preprocess.hic_process(hic_folder_path, output_path, resolution=1000, n_jobs=1)
    preprocess.hic_process(hic_folder_path, output_path, resolution=1000000, n_jobs=1)
