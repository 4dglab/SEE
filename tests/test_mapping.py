import pytest
from scce import preprocess, integrate


def test_hic_process():
    metadata_path = 'tests/data/hic/metadata.csv'
    hic_folder_path = 'tests/data/hic'
    output_path = 'tests/data/hic.h5'
    column_names = dict(id='sample_name', cell_type='cell_type')
    cell_types = ['Astro', 'Endo', 'ODC', 'OPC']
    resolution = 10000
    n_jobs = 8
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


def test_integrate():
    hic_path = 'tests/data/hic.h5'
    rna_path = 'tests/data/rna.h5'
    cell_types = ['Astro', 'Endo', 'Oligo', 'ODC', 'OPC']
    hic_pca_path = 'tests/data/other/hic_pca.csv'
    gtf_path, gtf_by = 'tests/data/other/gencode.v19.annotation.gtf', 'gene_name'
    resolution = 10000

    data_tool = integrate.DataTool(hic_path, rna_path, cell_types)
    data_tool.add_hic_pca(hic_pca_path)
    data_tool.add_gene_annotation(gtf_path, gtf_by)

    data_tool.hic_pca()
    data_tool.rna_pca()
    data_tool.rna_highly_variable_genes()
    hic, rna = data_tool.get_data()

    hic.var['chromStart'], hic.var['chromEnd'] = hic.var['start'], hic.var['end']
    graph = integrate.generate_graph(hic, rna, resolution)
    integrate.glue_embedding(hic, rna, graph, CPU_ONLY=True)
    map = integrate.mapping(hic, rna)

    hic.write('tests/data/hic_mapped.h5', compression='gzip')
    rna.write('tests/data/rna_mapped.h5', compression='gzip')
    map.to_csv('tests/data/map.csv')
