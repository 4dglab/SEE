conda create -n sclab python=3.8
conda install pytorch cudatoolkit=11.3 -c pytorch
pip install scglue
pip install --user scikit-misc
pip install cooler
pip install einops
pip install vit-pytorch

pip install fa2

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/train_dataset.npy -e /lmh_data/data/sclab/sclab/eval_dataset.npy -o /lmh_data/data/sclab/sclab/tmp/PDGFRA -g PDGFRA


python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/PDGFRA/model_epoch_9.pth -g PDGFRA -o /lmh_data/data/sclab/sclab/tmp/PDGFRA/evaluate.npy
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/SLC1A2/model_epoch_8.pth -g SLC1A2 -o /lmh_data/data/sclab/sclab/tmp/SLC1A2/evaluate.npy
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/MBP/model_epoch_9.pth -g MBP -o /lmh_data/data/sclab/sclab/tmp/MBP/evaluate.npy
python validate.py -e /lmh_data/data/sclab/sclab/AD/eval_dataset.npy -m /lmh_data/data/sclab/sclab/AD/tmp/SLC1A2/model_epoch_8.pth -g SLC1A2 -o /lmh_data/data/sclab/sclab/AD/tmp/SLC1A2/evaluate.npy

# pip install pyBigWig
# pip install --upgrade jupyter
# pip install ipykernel
# python -m ipykernel install --user --name sclab