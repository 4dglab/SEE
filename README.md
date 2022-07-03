conda create -n sclab python=3.8
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install scglue
<!-- conda install -c conda-forge -c bioconda scglue pytorch-gpu  # With GPU support -->

pip install cooler
pip install --user scikit-misc
pip install protobuf==3.20.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install -c pytorch faiss-gpu
<!-- pip install ipykernel -->
<!-- conda install -c conda-forge ipywidgets -->