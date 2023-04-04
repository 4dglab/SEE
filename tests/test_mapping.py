import pytest
from scce.mapping import preprocess


def test_hic_process():
    hic_folder_path = 'tests/data/hic'
    output_path = 'tests/data/hic.h5'
    with pytest.raises(ValueError, match=r".* resolution is not .*"):
        preprocess.hic_process(hic_folder_path, output_path, resolution=1000, n_jobs=1)
    preprocess.hic_process(hic_folder_path, output_path, resolution=1000000, n_jobs=1)

def test_rna_process():
    metadata_path = 'tests/data/rna/metadata.csv'
    matrix_path = 'tests/data/rna/matrix.csv'
    output_path = 'tests/data/rna.h5'
    column_names = dict(id='sample_name', cell_type='subclass_label')
    cell_types = ['Astro', 'Endo', 'Oligo', 'OPC']
    preprocess.rna_process(metadata_path, matrix_path, output_path, column_names, cell_types)
