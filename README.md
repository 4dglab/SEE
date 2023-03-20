# SEE
## Get the code
git clone https://github.com/LMH0066/SEE.git --depth=1
## Prepare the environment
The see environment can be installed via conda:
```
conda env create -f environment.yml
```
## Directory structure
.  
|-- README.md  
|-- analyse  
|   |-- 3DMax  
|   |   |-- 3DMax.jar  
|   |   |-- get_target_bulk_hic.py  
|   |   |-- get_volume.ipynb  
|   |   \`-- other folders(The result of 3D Max)  
|   |-- AD  
|   |   |-- AD&control hic \*.ipynb  
|   |   |-- create train dataset(specifies the location).py  
|   |   |-- create_dataset-scanorama.ipynb  
|   |   \`-- tmp  
|   |-- Data Analysis(PDGFRA).ipynb  
|   |-- Mapping Figure.ipynb  
|   |-- analyse_util.py  
|   |-- bulk  
|   |   |-- MBP.ipynb  
|   |   \`-- create_dataset-scanorama.ipynb  
|   |-- figure_file  
|   |-- loss-effectiveness  
|   |   |-- Pearsonr Analysis.ipynb  
|   |   \`-- Validate Accuracy Analysis.ipynb  
|   |-- quality  
|   |   |-- ARI.ipynb  
|   |   |-- Landmark position-\*.ipynb  
|   |   |-- PDGFRA-OPC_ODC.ipynb  
|   |   |-- PDGFRA.ipynb  
|   |   |-- SLC1A2.ipynb  
|   |   \`-- single vs bulk.ipynb  
|   |-- related-genes  
|   |   |-- Landmark position-\*.ipynb  
|   |   |-- Related Genes\*.ipynb  
|   |   |-- Scenic Astro/OPC/ODC.ipynb  
|   |   |-- Scenic Calculation.ipynb  
|   |   |-- differential related genes\*.ipynb  
|   |   |-- judgement structural genes.ipynb  
|   |   \`-- tmp  
|   \`-- velocity  
|       |-- 3D analysis under each pseudo-time (MBP).ipynb  
|       |-- Early  
|       |-- Late  
|       |-- MBP.ipynb  
|       \`-- PDGFRA.ipynb  
|-- environment.yml  
|-- script
|   |-- 1. scHiC data preprocess.ipynb  
|   |-- 2. scRNA data preprocess.ipynb  
|   |-- 3. Glue data preprocess.ipynb  
|   |-- 4. Glue model train.ipynb  
|   |-- 5. Mapping.ipynb  
|   |-- 6. marker gene catch.ipynb  
|   |-- 7. create train dataset(promoter).py  
|   |-- 7. create train dataset(terminator).py  
|   \`-- 7. create train dataset.py  
\`-- train  
    |-- dataset.py  
    |-- focalloss.py  
    |-- net.py  
    |-- train_model.py  
    |-- util.py  
    \`-- validate.py  

## Train
### train
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /folder/to/train_file -e /folder/to/eval_file -o /folder/to/output_folder -g gene_name
```
### validate
```
python validate.py -e /folder/to/eval_file -m /path/to/model -g gene_name -o /folder/to/output_file -s output_size
```
