# SEE
## Installation
```
conda create -n scce python=3.8 libffi=3.3
pip install scce
```
## Paper Content
### Get the code
```
git clone https://github.com/LMH0066/SEE.git --depth=1
```
### Prepare the environment
The see environment can be installed via conda:
```
conda env create -f environment.yml
```
### Directory structure
```
.
|-- script                            # Obtain training data through public data
|-- train                             # Main code for training model
|-- analyse                           # Experiments
|   |-- 3DMax                         # 
|   |-- AD                            # Case analysis of Alzheimer's disease
|   |-- Data Analysis(PDGFRA).ipynb   # Case study of raw data
|   |-- analyse_util.py               # Some common functions used in the analysis process
|   |-- bulk                          # Case analysis of bulk RNA
|   |-- loss-effectiveness            # FocalLoss effectiveness
|   |-- quality                       # Method evaluation
|   |-- related-genes                 # Importance analysis of input features
|   \`-- velocity                     # Case analysis of pseudo-time
|-- environment.yml
\`-- README.md
```
### Train
#### train
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_model.py -t /folder/to/train_file -e /folder/to/eval_file -o /folder/to/output_folder -g gene_name
```
#### validate
```
python validate.py -e /folder/to/eval_file -m /path/to/model -g gene_name -o /folder/to/output_file -s output_size
```
### Analyse
All the analysis results in the paper can be found in the code under the 'analyse' folder.
