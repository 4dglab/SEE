import pytest
from scce import preprocess, integrate


def test_hic_process():
    metadata_path = 'tests/data/hic/metadata.csv'
    hic_folder_path = 'tests/data/hic'
    output_path = 'tests/data/hic.h5'
    column_names = dict(id='sample_name', cell_type='cell_type')
    cell_types = ['Astro', 'Endo', 'Oligo', 'OPC']
    resolution = 1000000
    n_jobs = 1
    with pytest.raises(ValueError, match=r".* resolution is not .*"):
        preprocess.hic_process(metadata_path, hic_folder_path, output_path, column_names, cell_types, resolution=1000, n_jobs=n_jobs)
    with pytest.raises(ValueError, match=r"No valid sample in metadata."):
        preprocess.hic_process(metadata_path, hic_folder_path, output_path, column_names, ['XXX'], resolution=resolution, n_jobs=n_jobs)
    preprocess.hic_process(metadata_path, hic_folder_path, output_path, column_names, cell_types, resolution=resolution, n_jobs=n_jobs)

def test_rna_process():
    metadata_path = 'tests/data/rna/metadata.csv'
    matrix_path = 'tests/data/rna/matrix.csv'
    output_path = 'tests/data/rna.h5'
    column_names = dict(id='sample_name', cell_type='subclass_label')
    cell_types = ['Astro', 'Endo', 'Oligo', 'OPC']
    preprocess.rna_process(metadata_path, matrix_path, output_path, column_names, cell_types)

def test_glue_map():
    hic_path = 'tests/data/hic.h5'
    rna_path = 'tests/data/rna.h5'
    cell_types = ['Astro', 'Endo', 'Oligo', 'OPC']
    hic_pca_path = 'tests/data/hic_pca.csv'
    gtf_path, gtf_by = 'tests/data/gtf.gtf', 'gene_id'
    resolution = 10000

    data_tool = integrate.DataTool(hic_path, rna_path, cell_types)
    data_tool.add_hic_pca(hic_pca_path)
    data_tool.add_gene_annotation(gtf_path, gtf_by)

    data_tool.hic_pca()
    data_tool.rna_pca()
    data_tool.rna_highly_variable_genes()
    hic, rna = data_tool.get_data()

    graph = integrate.generate_graph(hic, rna, resolution)
    integrate.glue_embedding(hic, rna)
    map = integrate.mapping(hic, rna)
