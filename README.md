conda create -n sclab python=3.8
conda install pytorch cudatoolkit=11.3 -c pytorch
pip install scglue
pip install --user scikit-misc
pip install cooler
pip install einops
pip install vit-pytorch

pip install fa2

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_model.py -t /lmh_data/data/sclab/sclab/train_dataset.npy -e /lmh_data/data/sclab/sclab/eval_dataset.npy -o /lmh_data/data/sclab/sclab/tmp/PDGFRA -g PDGFRA
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/train_dataset.npy -e /lmh_data/data/sclab/sclab/eval_dataset.npy -o /lmh_data/data/sclab/sclab/tmp/PDGFRA_OPC_ODC -g PDGFRA

# pip install pyBigWig
# pip install --upgrade jupyter
# pip install ipykernel
# python -m ipykernel install --user --name sclab