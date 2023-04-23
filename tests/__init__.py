from os.path import join as opj

from scce import utils


class FileHelper:
    def __init__(self):
        self.output_folder_path = 'tests/data/output'
        utils.mkdir(self.output_folder_path)
    
    @property
    def preprocess_hic_path(self):
        return opj(self.output_folder_path, 'hic.h5')

    @property
    def preprocess_rna_path(self):
        return opj(self.output_folder_path, 'rna.h5')
    
    @property
    def mapped_hic_path(self):
        return opj(self.output_folder_path, 'hic_mapped.h5')

    @property
    def mapped_rna_path(self):
        return opj(self.output_folder_path, 'rna_mapped.h5')
    
    @property
    def mapping_path(self):
        return opj(self.output_folder_path, 'map.csv')
    
    @property
    def umap_path(self):
        return opj(self.output_folder_path, 'umap.png')
    
    @property
    def evaluate_path(self):
        return opj(self.output_folder_path, 'evaluate.npy')
