# SEE
## Directory structure
## Requirement Installation
The see environment can be installed via conda:
```
conda env create -f environment.yml
```


CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/train_dataset.npy -e /lmh_data/data/sclab/sclab/eval_dataset.npy -o /lmh_data/data/sclab/sclab/tmp/PDGFRA -g PDGFRA


python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/PDGFRA/model_epoch_9.pth -g PDGFRA -o /lmh_data/data/sclab/sclab/tmp/PDGFRA/evaluate.npy
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/SLC1A2/model_epoch_8.pth -g SLC1A2 -o /lmh_data/data/sclab/sclab/tmp/SLC1A2/evaluate.npy
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/MBP/model_epoch_9.pth -g MBP -o /lmh_data/data/sclab/sclab/tmp/MBP/evaluate.npy
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/GPM6A/model_epoch_12.pth -g GPM6A -o /lmh_data/data/sclab/sclab/tmp/GPM6A/evaluate.npy -s 741


python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/SLC1A3/model_epoch_9.pth -g SLC1A3 -o /lmh_data/data/sclab/sclab/tmp/SLC1A3/evaluate.npy -s 45
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/QKI/model_epoch_10.pth -g MBP -o /lmh_data/data/sclab/sclab/tmp/QKI/evaluate.npy


python validate.py -e /lmh_data/data/sclab/sclab/AD/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/SLC1A2/model_epoch_8.pth -g SLC1A2 -o /lmh_data/data/sclab/sclab/AD/tmp/SLC1A2/evaluate.npy -s 171
python validate.py -e /lmh_data/data/sclab/sclab/AD/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/SLC1A3/model_epoch_9.pth -g SLC1A3 -o /lmh_data/data/sclab/sclab/AD/tmp/SLC1A3/evaluate.npy -s 45
python validate.py -e /lmh_data/data/sclab/sclab/AD/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/GPM6A/model_epoch_12.pth -g GPM6A -o /lmh_data/data/sclab/sclab/AD/tmp/GPM6A/evaluate.npy -s 741

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/AD/CLU/train_dataset.npy -e /lmh_data/data/sclab/sclab/AD/CLU/eval_dataset.npy -o /lmh_data/data/sclab/sclab/AD/CLU/tmp -g chr8_27450000_27510000
python validate.py -e /lmh_data/data/sclab/sclab/AD/eval_dataset.npy -m /lmh_data/data/sclab/sclab/AD/CLU/tmp/model_epoch_11.pth -g chr8_27450000_27510000 -o /lmh_data/data/sclab/sclab/AD/CLU/evaluate.npy -s 21

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/AD/APP/train_dataset.npy -e /lmh_data/data/sclab/sclab/AD/APP/eval_dataset.npy -o /lmh_data/data/sclab/sclab/AD/APP/tmp -g chr21_27240000_27560000
python validate.py -e /lmh_data/data/sclab/sclab/AD/eval_dataset.npy -m /lmh_data/data/sclab/sclab/AD/APP/tmp/model_epoch_9.pth -g chr21_27240000_27560000 -o /lmh_data/data/sclab/sclab/AD/APP/evaluate.npy -s 528

<!-- mouse -->
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/mouse/train_dataset.npy -e /lmh_data/data/sclab/sclab/mouse/eval_dataset.npy -o /lmh_data/data/sclab/sclab/mouse/tmp/Slc1a2 -g Slc1a2
python validate.py -e /lmh_data/data/sclab/sclab/mouse/eval_dataset.npy -m /lmh_data/data/sclab/sclab/mouse/tmp/Slc1a2/model_epoch_27.pth -g Slc1a2 -o /lmh_data/data/sclab/sclab/mouse/tmp/Slc1a2/evaluate.npy -s 120

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/mouse/tmp/chr15_8630000_8800000/train_dataset.npy -e /lmh_data/data/sclab/sclab/mouse/tmp/chr15_8630000_8800000/eval_dataset.npy -o /lmh_data/data/sclab/sclab/mouse/tmp/chr15_8630000_8800000 -g 15_8630000_8800000
python validate.py -e /lmh_data/data/sclab/sclab/mouse/tmp/chr15_8630000_8800000/eval_dataset.npy -m /lmh_data/data/sclab/sclab/mouse/tmp/chr15_8630000_8800000/model_epoch_18.pth -g 15_8630000_8800000 -o /lmh_data/data/sclab/sclab/mouse/tmp/chr15_8630000_8800000/evaluate.npy -s 153

python validate.py -e /lmh_data/data/sclab/sclab/bulk/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/MBP/model_epoch_9.pth -g MBP -o /lmh_data/data/sclab/sclab/bulk/tmp/MBP/evaluate.npy -s 153