import argparse
import pathlib
import subprocess
import threading


dpdchrom_path = '/home/liminghong/work/repos/sclab/scHiC_analyse/DPDchrom/dpdchrom'
rst2mol2_path = '/home/liminghong/work/repos/sclab/scHiC_analyse/DPDchrom/py_scripts/rst2mol2.py'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=pathlib.Path, required=True)
    parser.add_argument("-o", "--output", dest="output", type=pathlib.Path, required=True)
    parser.add_argument("-t", "--temp", dest="temp", type=pathlib.Path, required=True)
    return parser.parse_args()


def run(cmd, cwd):
    p = subprocess.Popen(args=cmd, shell=True, cwd=cwd)
    p.wait()


def main(args):
    input_dir: pathlib.Path = args.input
    output_dir: pathlib.Path = args.output
    temp_dir: pathlib.Path = args.temp
    if not input_dir.exists() or not input_dir.is_dir():
        raise RuntimeError("input path error")
    
    proc = []
    for file_path in input_dir.rglob('*'):
        _file_name = file_path.name
        _temp_dir = temp_dir.joinpath(_file_name)
        if not _temp_dir.exists():
            _temp_dir.mkdir()

        mol_file_path = str(output_dir.joinpath(_file_name + '.mol2'))
        _cmd = ' '.join([dpdchrom_path, str(file_path.absolute()), '10000', '0'])
        _cmd += '; ' + ' '.join(['python', rst2mol2_path, '-i restart11.dat -o {}'.format(mol_file_path)])
        proc.append(threading.Thread(target=run, args=(_cmd, str(_temp_dir.absolute()))))

    [p.start() for p in proc]
    [p.join() for p in proc]
    

if __name__ == "__main__":
    main(parse_args())