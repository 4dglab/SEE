import os
import anndata
import cooler
import pandas as pd
from joblib import Parallel, delayed


def load_hics(folder_path, n_jobs=1):
    def load_hic(folder_path, file_name):
        pass

    def load_cooler(folder_path, file_name):
        c = cooler.Cooler(os.path.join(folder_path, file_name))
        contact = c.pixels(join=True)[:]
        contact = contact[contact['start1']!=contact['start2']]
        binsize, chromsizes = c.binsize, c.chromsizes

        _1 = contact.groupby(['chrom1', 'start1'])['count'].sum()
        _2 = contact.groupby(['chrom2', 'start2'])['count'].sum()
        _1.index.names = _2.index.names = ['chrom', 'start']
        _1, _2 = _1[_1!=0], _2[_2!=0]
        info = pd.concat([_1, _2], axis=1).fillna(0).sum(axis=1).sort_index()
        
        _indexs = set([(chrom, int(i * binsize))
                for chrom in chromsizes.keys()
                for i in range(int(chromsizes[chrom]/binsize)+1)])
        _indexs -= set(info.index)
        info = pd.concat([info, pd.Series([0]*len(_indexs), index=list(_indexs))]).sort_index()

        return info.to_frame().astype('float16').rename(columns={0:file_name})

    joblist = []
    for _, _, file_names in os.walk(folder_path, topdown=True):
        for file_name in file_names:
            _suffix = os.path.splitext(file_name)[-1]
            if _suffix not in ['.hic', '.cool']:
                print(f'File {file_name} is not a valid file.')
                continue

            _func = load_hic if _suffix == '.hic' else load_cooler
            joblist.append(delayed(_func)(folder_path, file_name))

    infos = Parallel(n_jobs=n_jobs, backend='loky', verbose=1)(joblist)
    infos = pd.concat(infos, axis=1).fillna(0).sort_index()
    return infos

def hic_process(hic_folder_path, output_path, n_jobs=1):
    infos = load_hics(hic_folder_path, n_jobs)

    obs = pd.DataFrame(infos.T.index, columns=['cells'])
    obs.insert(obs.shape[1] - 1, 'domain', 'scHiC')
    obs = obs.set_index('cells')
    var = infos.reset_index()[['chrom', 'start']].set_index(infos.index.map('{0[0]}_{0[1]}'.format))

    infos.index = infos.index.map('{0[0]}_{0[1]}'.format)
    infos = anndata.AnnData(X=infos.T, obs=obs, var=var)

    infos.write(output_path, compression='gzip')
