# should openjdk-8-jdk and 3DMax
import os
import sys
sys.path.append('/lmh_data/work/SEE/')

import numpy as np
import tempfile
from train.util import array2mat

train_set = np.load('/lmh_data/data/sclab/sclab/train_dataset.npy', allow_pickle=True)
eval_set = np.load('/lmh_data/data/sclab/sclab/eval_dataset.npy', allow_pickle=True)
data_set = np.append(train_set, eval_set)

# gene_name, cell_type = 'SLC1A2', 'Astro'
# gene_name, cell_type = 'QKI', 'ODC'
gene_name, cell_type = 'MBP', 'ODC'


def show_by_cell_type(cell_type):
    _hics, num = None, 0
    for _data in data_set:
        if _data['cell_type'] != cell_type:
            continue
        if _data['identity'] != 'truth':
            continue
        num += 1
        _hic = _data['scHiC'][gene_name].astype(float)
        if _hics is None:
            _hics = _hic.copy()
        else:
            _hics += _hic
        # if num >= 250:
            # break
    # if num < 250:
        # raise Exception()
    return _hics/num
mat = array2mat(show_by_cell_type(cell_type))
mat = mat / mat.max() * 1000

ftmp = tempfile.NamedTemporaryFile(delete=False)
print(ftmp.name)
with open(ftmp.name, 'w') as f:
    _strs = []
    for i in range(mat.shape[0]):
        _strs.append(','.join([str(j) for j in mat[i]])+'\n')
    f.writelines(_strs)

out_dir = gene_name + '_' + cell_type
os.makedirs(out_dir, exist_ok=True)
with open('parameters.txt', 'w') as f:
    f.writelines([
        'NUM = 1\n', 'OUTPUT_FOLDER = {}\n'.format(out_dir), 'INPUT_FILE = {}\n'.format(ftmp.name),
        'VERBOSE = true\n', 'LEARNING_RATE = 1\n', 'MAX_ITERATION = 10000\n'
    ])

os.system('java -jar 3DMax.jar parameters.txt')

os.remove(ftmp.name)
os.remove('parameters.txt')