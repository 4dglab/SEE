import os
import sys
import argparse

import anndata
import cooler
import pandas as pd
from joblib import Parallel, delayed


def main(args):
    input_folder_path = args.input_folder_path
    output_file_path = args.output_file_path
    parallel_nums = args.parallel_nums

    parallel = Parallel(n_jobs=parallel_nums, backend='loky', verbose=1)

    def load_coolers(folder_path):
        def load_cooler(folder_path, file_name):
            c = cooler.Cooler(os.path.join(folder_path, file_name))
            contact = c.pixels(join=True)[:]
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
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for file_name in files:
                joblist.append(delayed(load_cooler)(folder_path, file_name))

        infos = parallel(joblist)
        infos = pd.concat(infos, axis=1).fillna(0).sort_index()
        return infos
    infos = load_coolers(input_folder_path)

    obs = pd.DataFrame(infos.T.index, columns=['cells'])
    obs.insert(obs.shape[1] - 1, 'domain', 'scHiC')
    obs = obs.set_index('cells')
    var = infos.reset_index()[['chrom', 'start']].set_index(infos.index.map('{0[0]}_{0[1]}'.format))

    infos.index = infos.index.map('{0[0]}_{0[1]}'.format)
    infos = anndata.AnnData(X=infos.T, obs=obs, var=var)
    infos.write(output_file_path, compression="gzip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='coolers 2 anndata')
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-i', dest='input_folder_path', help='', required=True)
    req_args.add_argument('-o', dest='output_file_path', help='', required=True)
    req_args.add_argument('-n', dest='parallel_nums', type=int, help='', required=True)

    args = parser.parse_args(sys.argv[1:])
    main(args)