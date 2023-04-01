import os
import tempfile
import numpy as np


def Calculate_chromatin_conformation(mat, out_dir, numbers=5):
    for _, _, filenames in os.walk(out_dir, topdown=True):  
        for filename in filenames:
            if os.path.splitext(filename)[-1]=='.pdb':
                numbers -= 1
        break

    for i in range(numbers):
        ftmp = tempfile.NamedTemporaryFile(delete=False)
        with open(ftmp.name, 'w') as f:
            _strs = []
            for i in range(mat.shape[0]):
                _strs.append(','.join([str(j) for j in mat[i]])+'\n')
            f.writelines(_strs)

        os.makedirs(out_dir, exist_ok=True)
        with open('parameters.txt', 'w') as f:
            f.writelines([
                'NUM = 1\n', 'OUTPUT_FOLDER = {}\n'.format(out_dir), 'INPUT_FILE = {}\n'.format(ftmp.name),
                'VERBOSE = true\n', 'LEARNING_RATE = 1\n', 'MAX_ITERATION = 10000\n'
            ])

        os.system('java -jar /lmh_data/work/SEE/analyse/3DMax/3DMax.jar parameters.txt > /dev/null')

        os.remove(ftmp.name)
        os.remove('parameters.txt')


def find_best_pdb(_dir):
    _max_score, _max_score_pdb_path = 0, ''
    for _, _, filenames in os.walk(_dir, topdown=True):
        for filename in filenames:
            if not filename.endswith('_log.txt') or filename == 'best_alpha_log.txt':
                continue
            with open(os.path.join(_dir, filename), 'r') as f:
                _score = float(f.readlines()[5].split(': ')[-1])
            if _max_score < _score:
                _max_score = _score
                _max_score_pdb_path = os.path.join(_dir, [
                    _name for _name in filenames
                    if _name.startswith(filename.split('_log.txt')[0]) and _name.endswith('.pdb')
                ][0])
        break

    if _max_score == 0:
        raise Exception()
    return _max_score_pdb_path


def read_pdb(_path):
    with open(_path, 'r') as f:
        _datas = f.readlines()[1:]
        _datas = [_data for _data in _datas if _data.startswith('ATOM')]
        _datas = [_data.split()[5:8] for _data in _datas]
        
    return np.array(_datas).astype('float')