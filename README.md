# SEE
SEE is a method for predicting the dynamics of chromatin conformation based on single-cell gene expression.
## Usage
The input of SEE is scRNA-seq data for all genes, and the output is scHi-C information for **a single specific region**.  
For an efficient balance between computational load and accuracy, we recommend using SEE at the megabase scale or smaller.  
The entire framework is divided into preprocessing, integration, training, and analysis. Only the training stage requires users to specify the chromatin region, using one parameter.
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
## Cite
Cite our paper by
```
@article{li2025see,
    author = {Li, Minghong and Yang, Yurong and Wu, Rucheng and Gong, Haiyan and Yuan, Zan and Wang, Jixin and Long, Erping and Zhang, Xiaotong and Chen, Yang},
    title = {SEE: A Method for Predicting the Dynamics of Chromatin Conformation Based on Single-Cell Gene Expression},
    journal = {Advanced Science},
    pages = {2406413},
    year = {2025},
    doi = {https://doi.org/10.1002/advs.202406413},
    url = {https://advanced.onlinelibrary.wiley.com/doi/abs/10.1002/advs.202406413},
    eprint = {https://advanced.onlinelibrary.wiley.com/doi/pdf/10.1002/advs.202406413}
}
```