#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "ERROR! Illegal number of parameters. Usage: bash install.sh conda_install_path environment_name"
    exit 0
fi

conda_install_path=$1
conda_env_name=$2

source $conda_install_path/etc/profile.d/conda.sh
echo "****************** Creating conda environment ${conda_env_name} python=3.8.5 ******************"
conda create -y --name $conda_env_name python=3.8.5

echo ""
echo ""
echo "****************** Activating conda environment ${conda_env_name} ******************"
conda activate $conda_env_name

echo ""
echo ""
echo "****************** Installing pytorch with cuda10.2 ******************"
conda install -y pytorch torchvision cudatoolkit=10.2 -c pytorch

echo ""
echo ""
echo "****************** Installing matplotlib ******************"
conda install -y matplotlib

echo ""
echo ""
echo "****************** Installing opencv, scipy, tqdm, exifread ******************"
pip install opencv-python
pip install scipy
pip install tqdm
pip install exifread

echo ""
echo ""
echo "****************** Installing cupy (dependency for pwcnet) ******************"

conda install -y -c conda-forge cupy=7.8.0

echo ""
echo ""
echo "****************** Installing rawpy ******************"

pip install rawpy

echo ""
echo ""
echo "****************** Installing LPIPS ******************"

pip install lpips

echo ""
echo ""
echo "****************** Installing tensorboard ******************"

pip install tb-nightly

echo ""
echo ""
echo "****************** Installing jpeg4py python wrapper ******************"
pip install jpeg4py

echo ""
echo ""
echo "****************** Installing gdown ******************"
pip install gdown


echo ""
echo ""
echo "****************** Installing spatial correlation sampler (needed for custom flow networks) ******************"
pip install spatial_correlation_sampler

echo ""
echo ""
echo "****************** Setting up environment ******************"
python -c "from admin.environment import create_default_local_file; create_default_local_file()"

echo ""
echo ""
echo "****************** Installation complete! ******************"
