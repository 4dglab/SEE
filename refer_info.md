CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/train_dataset.npy -e /lmh_data/data/sclab/sclab/eval_dataset.npy -o /lmh_data/data/sclab/sclab/tmp/PDGFRA -g PDGFRA


python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/PDGFRA/model_epoch_9.pth -g PDGFRA -o /lmh_data/data/sclab/sclab/tmp/PDGFRA/evaluate.npy -s 36
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/SLC1A2/model_epoch_8.pth -g SLC1A2 -o /lmh_data/data/sclab/sclab/tmp/SLC1A2/evaluate.npy -s 171
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/MBP/model_epoch_9.pth -g MBP -o /lmh_data/data/sclab/sclab/tmp/MBP/evaluate.npy
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/GPM6A/model_epoch_12.pth -g GPM6A -o /lmh_data/data/sclab/sclab/tmp/GPM6A/evaluate.npy -s 741


python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/SLC1A3/model_epoch_9.pth -g SLC1A3 -o /lmh_data/data/sclab/sclab/tmp/SLC1A3/evaluate.npy -s 45
python validate.py -e /lmh_data/data/sclab/sclab/eval_dataset.npy -m /lmh_data/data/sclab/sclab/tmp/QKI/model_epoch_10.pth -g MBP -o /lmh_data/data/sclab/sclab/tmp/QKI/evaluate.npy

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

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/GSE223917/train_dataset.npy -e /lmh_data/data/sclab/GSE223917/eval_dataset.npy -o /lmh_data/data/sclab/GSE223917/tmp/Bcl11b -g Bcl11b
python validate.py -e /lmh_data/data/sclab/GSE223917/eval_dataset.npy -m /lmh_data/data/sclab/GSE223917/tmp/Bcl11b/model_epoch_12.pth -g Bcl11b -o /lmh_data/data/sclab/GSE223917/tmp/Bcl11b/evaluate.npy -s 55


CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/AD/DLC1/train_dataset.npy -e /lmh_data/data/sclab/sclab/AD/DLC1/eval_dataset.npy -o /lmh_data/data/sclab/sclab/AD/DLC1/tmp -g chr8_12940000_13470000
python validate.py -e /lmh_data/data/sclab/sclab/AD/eval_dataset.npy -m /lmh_data/data/sclab/sclab/AD/DLC1/tmp/model_epoch_10.pth -g chr8_12940000_13470000 -o /lmh_data/data/sclab/sclab/AD/DLC1/AD_evaluate.npy -s 1431
python validate.py -e /lmh_data/data/sclab/sclab/AD/DLC1/train_dataset.npy -m /lmh_data/data/sclab/sclab/AD/DLC1/tmp/model_epoch_10.pth -g chr8_12940000_13470000 -o /lmh_data/data/sclab/sclab/AD/DLC1/train_data_evaluate.npy -s 1431

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/AD/BIN1/train_dataset.npy -e /lmh_data/data/sclab/sclab/AD/BIN1/eval_dataset.npy -o /lmh_data/data/sclab/sclab/AD/BIN1/tmp -g chr2_127800000_128000000
python validate.py -e /lmh_data/data/sclab/sclab/AD/eval_dataset.npy -m /lmh_data/data/sclab/sclab/AD/BIN1/tmp/model_epoch_6.pth -g chr2_127800000_128000000 -o /lmh_data/data/sclab/sclab/AD/BIN1/AD_evaluate.npy -s 210
python validate.py -e /lmh_data/data/sclab/sclab/AD/BIN1/train_dataset.npy -m /lmh_data/data/sclab/sclab/AD/BIN1/tmp/model_epoch_6.pth -g chr2_127800000_128000000 -o /lmh_data/data/sclab/sclab/AD/BIN1/train_data_evaluate.npy -s 210

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/AD/CLU/train_dataset.npy -e /lmh_data/data/sclab/sclab/AD/CLU/eval_dataset.npy -o /lmh_data/data/sclab/sclab/AD/CLU/tmp -g chr8_27400000_27500000
python validate.py -e /lmh_data/data/sclab/sclab/AD/eval_dataset.npy -m /lmh_data/data/sclab/sclab/AD/CLU/tmp/model_epoch_12.pth -g chr8_27400000_27500000 -o /lmh_data/data/sclab/sclab/AD/CLU/AD_evaluate.npy -s 55
python validate.py -e /lmh_data/data/sclab/sclab/AD/CLU/train_dataset.npy -m /lmh_data/data/sclab/sclab/AD/CLU/tmp/model_epoch_12.pth -g chr8_27400000_27500000 -o /lmh_data/data/sclab/sclab/AD/CLU/train_data_evaluate.npy -s 55

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /lmh_data/data/sclab/sclab/AD/KAT8/train_dataset.npy -e /lmh_data/data/sclab/sclab/AD/KAT8/eval_dataset.npy -o /lmh_data/data/sclab/sclab/AD/KAT8/tmp -g chr16_31000000_31240000
python validate.py -e /lmh_data/data/sclab/sclab/AD/eval_dataset.npy -m /lmh_data/data/sclab/sclab/AD/KAT8/tmp/model_epoch_11.pth -g chr16_31000000_31240000 -o /lmh_data/data/sclab/sclab/AD/KAT8/AD_evaluate.npy -s 300
python validate.py -e /lmh_data/data/sclab/sclab/AD/KAT8/train_dataset.npy -m /lmh_data/data/sclab/sclab/AD/KAT8/tmp/model_epoch_11.pth -g chr16_31000000_31240000 -o /lmh_data/data/sclab/sclab/AD/KAT8/train_data_evaluate.npy -s 300

<!-- random test -->
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /data/liminghong/sclab/sclab/random_test/train_dataset.npy -e /data/liminghong/sclab/sclab/random_test/eval_dataset.npy -o /data/liminghong/sclab/sclab/random_test/tmp/chr6_152230000_152590000 -g chr6_152230000_152590000
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 train_model.py -t /data/liminghong/sclab/sclab/random_test/train_dataset.npy -e /data/liminghong/sclab/sclab/random_test/eval_dataset.npy -o /data/liminghong/sclab/sclab/random_test/tmp/chr5_41480000_41840000 -g chr5_41480000_41840000
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29502 train_model.py -t /data/liminghong/sclab/sclab/random_test/train_dataset.npy -e /data/liminghong/sclab/sclab/random_test/eval_dataset.npy -o /data/liminghong/sclab/sclab/random_test/tmp/chr21_20260000_20500000 -g chr21_20260000_20500000
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29503 train_model.py -t /data/liminghong/sclab/sclab/random_test/train_dataset.npy -e /data/liminghong/sclab/sclab/random_test/eval_dataset.npy -o /data/liminghong/sclab/sclab/random_test/tmp/chr2_176640000_176840000 -g chr2_176640000_176840000
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29504 train_model.py -t /data/liminghong/sclab/sclab/random_test/train_dataset.npy -e /data/liminghong/sclab/sclab/random_test/eval_dataset.npy -o /data/liminghong/sclab/sclab/random_test/tmp/chr4_69390000_69680000 -g chr4_69390000_69680000
CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29505 train_model.py -t /data/liminghong/sclab/sclab/random_test/train_dataset.npy -e /data/liminghong/sclab/sclab/random_test/eval_dataset.npy -o /data/liminghong/sclab/sclab/random_test/tmp/chr13_71100000_71370000 -g chr13_71100000_71370000

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /data/liminghong/sclab/sclab/random_test/train_dataset.npy -e /data/liminghong/sclab/sclab/random_test/eval_dataset.npy -o /data/liminghong/sclab/sclab/random_test/tmp/chr17_14790000_15060000 -g chr17_14790000_15060000
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 train_model.py -t /data/liminghong/sclab/sclab/random_test/train_dataset.npy -e /data/liminghong/sclab/sclab/random_test/eval_dataset.npy -o /data/liminghong/sclab/sclab/random_test/tmp/chr20_51150000_51590000 -g chr20_51150000_51590000
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29502 train_model.py -t /data/liminghong/sclab/sclab/random_test/train_dataset.npy -e /data/liminghong/sclab/sclab/random_test/eval_dataset.npy -o /data/liminghong/sclab/sclab/random_test/tmp/chr14_50770000_50990000 -g chr14_50770000_50990000
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29503 train_model.py -t /data/liminghong/sclab/sclab/random_test/train_dataset.npy -e /data/liminghong/sclab/sclab/random_test/eval_dataset.npy -o /data/liminghong/sclab/sclab/random_test/tmp/chr1_227020000_227410000 -g chr1_227020000_227410000
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29504 train_model.py -t /data/liminghong/sclab/sclab/random_test/train_dataset.npy -e /data/liminghong/sclab/sclab/random_test/eval_dataset.npy -o /data/liminghong/sclab/sclab/random_test/tmp/chr12_118040000_118290000 -g chr12_118040000_118290000
CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29505 train_model.py -t /data/liminghong/sclab/sclab/random_test/train_dataset.npy -e /data/liminghong/sclab/sclab/random_test/eval_dataset.npy -o /data/liminghong/sclab/sclab/random_test/tmp/chr7_70290000_70540000 -g chr7_70290000_70540000


CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /data/liminghong/sclab/sclab/train_dataset.npy -e /data/liminghong/sclab/sclab/eval_dataset.npy -o /data/liminghong/sclab/sclab/tmp/PTPRZ1 -g PTPRZ1
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /data/liminghong/sclab/sclab/train_dataset.npy -e /data/liminghong/sclab/sclab/eval_dataset.npy -o /data/liminghong/sclab/sclab/tmp/VCAN -g VCAN
python validate.py -e /data/liminghong/sclab/sclab/eval_dataset.npy -m /data/liminghong/sclab/sclab/tmp/VCAN/model_epoch_13.pth -g VCAN -o /data/liminghong/sclab/sclab/tmp/VCAN/evaluate.npy -s 78


grep "PDGFRA" /data/liminghong/sclab/public/gencode.v19.annotation.gtf | awk '/\texon\t/' > /data/liminghong/sclab/sclab/random_test/SUPPA/PDGFRA.gtf
grep "PTPRZ1" /data/liminghong/sclab/public/gencode.v19.annotation.gtf | awk '/\texon\t/' > /data/liminghong/sclab/sclab/random_test/SUPPA/PTPRZ1.gtf
grep "VCAN" /data/liminghong/sclab/public/gencode.v19.annotation.gtf | awk '/\texon\t/' > /data/liminghong/sclab/sclab/random_test/SUPPA/VCAN.gtf
grep "SLC1A2" /data/liminghong/sclab/public/gencode.v19.annotation.gtf | awk '/\texon\t/' > /data/liminghong/sclab/sclab/random_test/SUPPA/SLC1A2.gtf
grep "SLC1A3" /data/liminghong/sclab/public/gencode.v19.annotation.gtf | awk '/\texon\t/' > /data/liminghong/sclab/sclab/random_test/SUPPA/SLC1A3.gtf
grep "GPM6A" /data/liminghong/sclab/public/gencode.v19.annotation.gtf | awk '/\texon\t/' > /data/liminghong/sclab/sclab/random_test/SUPPA/GPM6A.gtf
grep "MBP" /data/liminghong/sclab/public/gencode.v19.annotation.gtf | awk '/\texon\t/' > /data/liminghong/sclab/sclab/random_test/SUPPA/MBP.gtf
grep "QKI" /data/liminghong/sclab/public/gencode.v19.annotation.gtf | awk '/\texon\t/' > /data/liminghong/sclab/sclab/random_test/SUPPA/QKI.gtf
grep "DOCK10" /data/liminghong/sclab/public/gencode.v19.annotation.gtf | awk '/\texon\t/' > /data/liminghong/sclab/sclab/random_test/SUPPA/DOCK10.gtf
grep "PIP4K2A" /data/liminghong/sclab/public/gencode.v19.annotation.gtf | awk '/\texon\t/' > /data/liminghong/sclab/sclab/random_test/SUPPA/PIP4K2A.gtf

python suppa.py generateEvents -i /data/liminghong/sclab/sclab/random_test/SUPPA/PDGFRA.gtf -o /data/liminghong/sclab/sclab/random_test/SUPPA/PDGFRA -f ioe -e SE SS MX RI FL
python suppa.py generateEvents -i /data/liminghong/sclab/sclab/random_test/SUPPA/PTPRZ1.gtf -o /data/liminghong/sclab/sclab/random_test/SUPPA/PTPRZ1 -f ioe -e SE SS MX RI FL
python suppa.py generateEvents -i /data/liminghong/sclab/sclab/random_test/SUPPA/VCAN.gtf -o /data/liminghong/sclab/sclab/random_test/SUPPA/VCAN -f ioe -e SE SS MX RI FL
python suppa.py generateEvents -i /data/liminghong/sclab/sclab/random_test/SUPPA/SLC1A2.gtf -o /data/liminghong/sclab/sclab/random_test/SUPPA/SLC1A2 -f ioe -e SE SS MX RI FL
python suppa.py generateEvents -i /data/liminghong/sclab/sclab/random_test/SUPPA/SLC1A3.gtf -o /data/liminghong/sclab/sclab/random_test/SUPPA/SLC1A3 -f ioe -e SE SS MX RI FL
python suppa.py generateEvents -i /data/liminghong/sclab/sclab/random_test/SUPPA/GPM6A.gtf -o /data/liminghong/sclab/sclab/random_test/SUPPA/GPM6A -f ioe -e SE SS MX RI FL
python suppa.py generateEvents -i /data/liminghong/sclab/sclab/random_test/SUPPA/MBP.gtf -o /data/liminghong/sclab/sclab/random_test/SUPPA/MBP -f ioe -e SE SS MX RI FL
python suppa.py generateEvents -i /data/liminghong/sclab/sclab/random_test/SUPPA/QKI.gtf -o /data/liminghong/sclab/sclab/random_test/SUPPA/QKI -f ioe -e SE SS MX RI FL
python suppa.py generateEvents -i /data/liminghong/sclab/sclab/random_test/SUPPA/DOCK10.gtf -o /data/liminghong/sclab/sclab/random_test/SUPPA/DOCK10 -f ioe -e SE SS MX RI FL
python suppa.py generateEvents -i /data/liminghong/sclab/sclab/random_test/SUPPA/PIP4K2A.gtf -o /data/liminghong/sclab/sclab/random_test/SUPPA/PIP4K2A -f ioe -e SE SS MX RI FL


CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /data/liminghong/sclab/sclab/random_test/hk/train_dataset.npy -e /data/liminghong/sclab/sclab/random_test/hk/eval_dataset.npy -o /data/liminghong/sclab/sclab/random_test/hk/tmp/chr12_94071151_94288616 -g chr12_94071151_94288616
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 train_model.py -t /data/liminghong/sclab/sclab/random_test/hk/train_dataset.npy -e /data/liminghong/sclab/sclab/random_test/hk/eval_dataset.npy -o /data/liminghong/sclab/sclab/random_test/hk/tmp/chr6_37787275_38122400 -g chr6_37787275_38122400
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29502 train_model.py -t /data/liminghong/sclab/sclab/random_test/hk/train_dataset.npy -e /data/liminghong/sclab/sclab/random_test/hk/eval_dataset.npy -o /data/liminghong/sclab/sclab/random_test/hk/tmp/chr4_160025330_160281321 -g chr4_160025330_160281321
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29503 train_model.py -t /data/liminghong/sclab/sclab/random_test/hk/train_dataset.npy -e /data/liminghong/sclab/sclab/random_test/hk/eval_dataset.npy -o /data/liminghong/sclab/sclab/random_test/hk/tmp/chr11_43577986_43878167 -g chr11_43577986_43878167
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29504 train_model.py -t /data/liminghong/sclab/sclab/random_test/hk/train_dataset.npy -e /data/liminghong/sclab/sclab/random_test/hk/eval_dataset.npy -o /data/liminghong/sclab/sclab/random_test/hk/tmp/chr8_9413424_9639856 -g chr8_9413424_9639856
CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29505 train_model.py -t /data/liminghong/sclab/sclab/random_test/hk/train_dataset.npy -e /data/liminghong/sclab/sclab/random_test/hk/eval_dataset.npy -o /data/liminghong/sclab/sclab/random_test/hk/tmp/chr4_123073488_123283913 -g chr4_123073488_123283913

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /data/liminghong/sclab/sclab/random_test/hk/train_dataset.npy -e /data/liminghong/sclab/sclab/random_test/hk/eval_dataset.npy -o /data/liminghong/sclab/sclab/random_test/hk/tmp/chr2_98372799_98612388 -g chr2_98372799_98612388
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 train_model.py -t /data/liminghong/sclab/sclab/random_test/hk/train_dataset.npy -e /data/liminghong/sclab/sclab/random_test/hk/eval_dataset.npy -o /data/liminghong/sclab/sclab/random_test/hk/tmp/chr1_39546988_39952849 -g chr1_39546988_39952849
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29502 train_model.py -t /data/liminghong/sclab/sclab/random_test/hk/train_dataset.npy -e /data/liminghong/sclab/sclab/random_test/hk/eval_dataset.npy -o /data/liminghong/sclab/sclab/random_test/hk/tmp/chr3_47892182_48130769 -g chr3_47892182_48130769
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29503 train_model.py -t /data/liminghong/sclab/sclab/random_test/hk/train_dataset.npy -e /data/liminghong/sclab/sclab/random_test/hk/eval_dataset.npy -o /data/liminghong/sclab/sclab/random_test/hk/tmp/chr11_31833939_32127301 -g chr11_31833939_32127301
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29504 train_model.py -t /data/liminghong/sclab/sclab/random_test/hk/train_dataset.npy -e /data/liminghong/sclab/sclab/random_test/hk/eval_dataset.npy -o /data/liminghong/sclab/sclab/random_test/hk/tmp/chr10_80828792_81076276 -g chr10_80828792_81076276
CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29505 train_model.py -t /data/liminghong/sclab/sclab/random_test/hk/train_dataset.npy -e /data/liminghong/sclab/sclab/random_test/hk/eval_dataset.npy -o /data/liminghong/sclab/sclab/random_test/hk/tmp/chr20_34894258_35157040 -g chr20_34894258_35157040