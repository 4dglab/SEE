CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/train_dataset.npy -e /lmh_data/data/sclab/sclab/eval_dataset.npy -o /lmh_data/data/sclab/sclab/tmp/PDGFRA -g PDGFRA


python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/PDGFRA/model_epoch_9.pth -g PDGFRA -o /lmh_data/data/sclab/sclab/tmp/PDGFRA/evaluate.npy -s 36
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

<!-- fig4 -->
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/train_dataset.npy -e /lmh_data/data/sclab/sclab/eval_dataset.npy -o /lmh_data/data/sclab/sclab/tmp/CTNNA3 -g CTNNA3
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/CTNNA3/model_epoch_14.pth -g CTNNA3 -o /lmh_data/data/sclab/sclab/tmp/CTNNA3/evaluate.npy -s 153

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/train_dataset.npy -e /lmh_data/data/sclab/sclab/eval_dataset.npy -o /lmh_data/data/sclab/sclab/tmp/ST18 -g ST18
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/ST18/model_epoch_14.pth -g ST18 -o /lmh_data/data/sclab/sclab/tmp/ST18/evaluate.npy -s 666

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/train_dataset.npy -e /lmh_data/data/sclab/sclab/eval_dataset.npy -o /lmh_data/data/sclab/sclab/tmp/SLC44A1 -g SLC44A1
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/SLC44A1/model_epoch_8.pth -g SLC44A1 -o /lmh_data/data/sclab/sclab/tmp/SLC44A1/evaluate.npy -s 231

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/train_dataset.npy -e /lmh_data/data/sclab/sclab/eval_dataset.npy -o /lmh_data/data/sclab/sclab/tmp/RNF220 -g RNF220
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/RNF220/model_epoch_8.pth -g RNF220 -o /lmh_data/data/sclab/sclab/tmp/RNF220/evaluate.npy -s 351

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/train_dataset.npy -e /lmh_data/data/sclab/sclab/eval_dataset.npy -o /lmh_data/data/sclab/sclab/tmp/DPYD -g DPYD
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/DPYD/model_epoch_14.pth -g DPYD -o /lmh_data/data/sclab/sclab/tmp/DPYD/evaluate.npy -s 351

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/train_dataset.npy -e /lmh_data/data/sclab/sclab/eval_dataset.npy -o /lmh_data/data/sclab/sclab/tmp/PIP4K2A -g PIP4K2A
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/PIP4K2A/model_epoch_11.pth -g PIP4K2A -o /lmh_data/data/sclab/sclab/tmp/PIP4K2A/evaluate.npy -s 190

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/train_dataset.npy -e /lmh_data/data/sclab/sclab/eval_dataset.npy -o /lmh_data/data/sclab/sclab/tmp/NCKAP5 -g NCKAP5
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/NCKAP5/model_epoch_12.pth -g NCKAP5 -o /lmh_data/data/sclab/sclab/tmp/NCKAP5/evaluate.npy -s 4278

python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/ENPP2/model_epoch_9.pth -g ENPP2 -o /lmh_data/data/sclab/sclab/tmp/ENPP2/evaluate.npy -s 105

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/train_dataset.npy -e /lmh_data/data/sclab/sclab/eval_dataset.npy -o /lmh_data/data/sclab/sclab/tmp/PHLPP1 -g PHLPP1
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/PHLPP1/model_epoch_8.pth -g PHLPP1 -o /lmh_data/data/sclab/sclab/tmp/PHLPP1/evaluate.npy -s 406

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/train_dataset.npy -e /lmh_data/data/sclab/sclab/eval_dataset.npy -o /lmh_data/data/sclab/sclab/tmp/ELMO1 -g ELMO1
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/ELMO1/model_epoch_12.pth -g ELMO1 -o /lmh_data/data/sclab/sclab/tmp/ELMO1/evaluate.npy -s 1891

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/train_dataset.npy -e /lmh_data/data/sclab/sclab/eval_dataset.npy -o /lmh_data/data/sclab/sclab/tmp/PTPRK -g PTPRK
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/PTPRK/model_epoch_11.pth -g PTPRK -o /lmh_data/data/sclab/sclab/tmp/PTPRK/evaluate.npy -s 1653

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/train_dataset.npy -e /lmh_data/data/sclab/sclab/eval_dataset.npy -o /lmh_data/data/sclab/sclab/tmp/DOCK10 -g DOCK10
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/DOCK10/model_epoch_11.pth -g DOCK10 -o /lmh_data/data/sclab/sclab/tmp/DOCK10/evaluate.npy -s 465

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/train_dataset.npy -e /lmh_data/data/sclab/sclab/eval_dataset.npy -o /lmh_data/data/sclab/sclab/tmp/PCDH9 -g PCDH9
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/PCDH9/model_epoch_14.pth -g PCDH9 -o /lmh_data/data/sclab/sclab/tmp/PCDH9/evaluate.npy -s 4465

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/train_dataset.npy -e /lmh_data/data/sclab/sclab/eval_dataset.npy -o /lmh_data/data/sclab/sclab/tmp/PDE4B -g PDE4B
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/PDE4B/model_epoch_14.pth -g PDE4B -o /lmh_data/data/sclab/sclab/tmp/PDE4B/evaluate.npy -s 1830

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/train_dataset.npy -e /lmh_data/data/sclab/sclab/eval_dataset.npy -o /lmh_data/data/sclab/sclab/tmp/MOBP -g MOBP
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/MOBP/model_epoch_8.pth -g MOBP -o /lmh_data/data/sclab/sclab/tmp/MOBP/evaluate.npy -s 36

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/train_dataset.npy -e /lmh_data/data/sclab/sclab/eval_dataset.npy -o /lmh_data/data/sclab/sclab/tmp/MAN2A1 -g MAN2A1
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/MAN2A1/model_epoch_9.pth -g MAN2A1 -o /lmh_data/data/sclab/sclab/tmp/MAN2A1/evaluate.npy -s 190

<!-- fig4 Mb -->
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/Mb/train_dataset.npy -e /lmh_data/data/sclab/sclab/Mb/eval_dataset.npy -o /lmh_data/data/sclab/sclab/Mb/tmp/MBP -g MBP
python validate.py -e /lmh_data/data/sclab/sclab/Mb/eval_dataset.npy -m /lmh_data/data/sclab/sclab/Mb/tmp/MBP/model_epoch_13.pth -g MBP -o /lmh_data/data/sclab/sclab/Mb/tmp/MBP/evaluate.npy -s 6903

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/Mb/train_dataset.npy -e /lmh_data/data/sclab/sclab/Mb/eval_dataset.npy -o /lmh_data/data/sclab/sclab/Mb/tmp/CTNNA3 -g CTNNA3
python validate.py -e /lmh_data/data/sclab/sclab/Mb/eval_dataset.npy -m /lmh_data/data/sclab/sclab/Mb/tmp/CTNNA3/model_epoch_18.pth -g CTNNA3 -o /lmh_data/data/sclab/sclab/Mb/tmp/CTNNA3/evaluate.npy -s 39340

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/Mb/train_dataset.npy -e /lmh_data/data/sclab/sclab/Mb/eval_dataset.npy -o /lmh_data/data/sclab/sclab/Mb/tmp/ST18 -g ST18
python validate.py -e /lmh_data/data/sclab/sclab/Mb/eval_dataset.npy -m /lmh_data/data/sclab/sclab/Mb/tmp/ST18/model_epoch_18.pth -g ST18 -o /lmh_data/data/sclab/sclab/Mb/tmp/ST18/evaluate.npy -s 9316

python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/PDGFRA_Astro/model_epoch_12.pth -g PDGFRA -o /lmh_data/data/sclab/sclab/tmp/PDGFRA_Astro/evaluate.npy -s 36
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/PDGFRA_Astro_ODC/model_epoch_9.pth -g PDGFRA -o /lmh_data/data/sclab/sclab/tmp/PDGFRA_Astro_ODC/evaluate.npy -s 36
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/PDGFRA_Astro_OPC/model_epoch_19.pth -g PDGFRA -o /lmh_data/data/sclab/sclab/tmp/PDGFRA_Astro_OPC/evaluate.npy -s 36
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/PDGFRA_ODC/model_epoch_7.pth -g PDGFRA -o /lmh_data/data/sclab/sclab/tmp/PDGFRA_ODC/evaluate.npy -s 36
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/PDGFRA_OPC/model_epoch_6.pth -g PDGFRA -o /lmh_data/data/sclab/sclab/tmp/PDGFRA_OPC/evaluate.npy -s 36