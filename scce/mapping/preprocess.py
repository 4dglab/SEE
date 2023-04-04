import os
import anndata
import fanc
import pandas as pd
from joblib import Parallel, delayed


def load_hics(folder_path, resolution, n_jobs=1):
    def _load_hic(folder_path, file_name):
        _suffix = os.path.splitext(file_name)[-1]

        _file_path = os.path.join(folder_path, file_name)
        if _suffix in ['.hic', '.mcool']:
            c = fanc.load('{}@{}'.format(_file_path, resolution), mode='r')
        elif _suffix == '.cool':
            c = fanc.load(_file_path, mode='r')
        else:
            raise ValueError('File {file_name} is not a valid hic file.')
        
        binsize, chromosome_lengths = c.binsize, c.chromosome_lengths
        if _suffix == '.hic':
            # TODO
            pass
        else:
            contact = c.pixels(join=True)[:]

        if binsize != resolution:
            raise ValueError(f'File {file_name} resolution is not {resolution}.')

        return _get_contact(contact, binsize, chromosome_lengths).rename(columns={0:file_name})

    def _get_contact(contact, binsize, chromosome_lengths):
        contact = contact[contact['start1']!=contact['start2']]
        contact = contact[contact['chrom1']==contact['chrom2']]

        _1 = contact.groupby(['chrom1', 'start1'])['count'].sum()
        _2 = contact.groupby(['chrom2', 'start2'])['count'].sum()
        _1.index.names = _2.index.names = ['chrom', 'start']
        _1, _2 = _1[_1!=0], _2[_2!=0]
        info = pd.concat([_1, _2], axis=1).fillna(0).sum(axis=1).sort_index()
        
        _indexs = set([(chrom, int(i * binsize))
                for chrom in chromosome_lengths.keys()
                for i in range(int(chromosome_lengths[chrom]/binsize)+1)])
        _indexs -= set(info.index)
        info = pd.concat([info, pd.Series([0]*len(_indexs), index=list(_indexs))]).sort_index()

        return info.to_frame().astype('float16')

    joblist = [
        delayed(_load_hic)(folder_path, file_name)
        for _, _, file_names in os.walk(folder_path, topdown=True)
        for file_name in file_names
    ]

    infos = Parallel(n_jobs=n_jobs, backend='loky', verbose=1)(joblist)
    infos = pd.concat(infos, axis=1).fillna(0).sort_index()
    return infos

def hic_process(hic_folder_path, output_path, resolution=10000, n_jobs=1):
    infos = load_hics(hic_folder_path, resolution, n_jobs)

    obs = pd.DataFrame(infos.T.index, columns=['cells'])
    obs.insert(obs.shape[1] - 1, 'domain', 'scHiC')
    obs = obs.set_index('cells')
    var = infos.reset_index()[['chrom', 'start']].set_index(infos.index.map('{0[0]}_{0[1]}'.format))

    infos.index = infos.index.map('{0[0]}_{0[1]}'.format)
    infos = anndata.AnnData(X=infos.T, obs=obs, var=var, dtype='float16')

    infos.write(output_path, compression='gzip')

def rna_process(
        metadata_path, matrix_path, output_path,
        column_names=dict(id='sample_name', cell_type='cell_type'), cell_types=[]
    ):
    sample_name, cell_type = column_names['id'], column_names['cell_type']

    _metadata = pd.read_csv(metadata_path)
    if cell_types:
        _metadata = _metadata[_metadata[cell_type].isin(cell_types)]

    infos = pd.DataFrame()
    for chunk in pd.read_csv(matrix_path, chunksize=10000):
        _filter = chunk[chunk[sample_name].isin(_metadata[sample_name])]
        infos = pd.concat([infos, _filter])
    infos = infos.set_index(sample_name)
    infos = anndata.AnnData(X=infos, dtype='int32')

    _metadata = _metadata.set_index(sample_name)
    _metadata = _metadata.loc[infos.obs.index]
    infos.obs['cell_type'] = _metadata[cell_type]
    infos.obs['domain'] = 'scRNA'

    infos.write(output_path, compression="gzip")
