# Benchmarking Burst Super-Resolution for Polarization Images: Noise Dataset and Analysis
### [Project page](https://vclab.kaist.ac.kr/iccv2025p1/index.html) | [Paper](https://vclab.kaist.ac.kr/iccv2025p1/polar-denoising-main.pdf) | [Supplemental document](https://vclab.kaist.ac.kr/iccv2025p1/polar-denoising-supple.pdf) | [Datasets](https://drive.google.com/drive/folders/16z4gJCeky2frSAVkGZwxOEIVvqymqqbD?usp=drive_link)

[Inseung Hwang](https://sites.google.com/view/inseunghwang),
[Kiseok Choi](https://sites.google.com/view/kiseokchoi),
[Hyunho Ha](https://sites.google.com/view/hyunhoha),
[Min H. Kim](http://vclab.kaist.ac.kr/minhkim/)
<br>
KAIST
<br>
In this repository, we provide the code and datasets for the paper 'Benchmarking Burst Super-Resolution for Polarization Images: Noise Dataset and Analysis'
, reviewed and presented at **ICCV 2025**.

# Contents
1. Noise analysis using PolarNS dataset (MATLAB)
2. Polarization-trained burst super-resolution (Python)

# Noise analysis using PolarNS dataset (MATLAB)
The **PolarNS dataset** consists of 244 scenes, providing noise-reduced ground truth, noise statistics, and analysis results.

## Structure of dataset
The dataset is organized as follows:
* ```/histograms```: The output files of noise analysis 
* ```/object```: The images statistics from objects in the darkroom
  * ```/capture_info.txt```: The camera settings when the images are captured
  * ```/img_stat.mat```: The mean and standard deviation of the scene. The pixel values of raw images are in [0, 4095] range. 
* ```/scene```: The images statistics from scenes indoor and outdoor.
  * ```/capture_info.txt```
  * ```/img_stat.mat```
 
## Reproducing Figures & Tables
The analysis codes are located in ```/codes/matlab/```. Please ensure the path to the PolarNS dataset is correctly set in the main directory script before running.
You can reproduce the figures and tables from the paper using the following scripts:
|Paper Component|Description|Executable codes|
|------|---|---|
|Figure 3|noise analysis model|```main_calc_pdf.m```|
|Figure 4|noise analysis model validation|```main_figures_from_histogram.m```|
|Table 2, Figure 6|noise analysis|```main_figures_from_statistics_histogram.m```|
|Supp Figure 1.|Stokes vector noise model| ```main_figures_from_stokes_histogram.m```|

# Polarization-trained Burst Super-Resolution (Python)
We provide datasets for training and testing burst super-resolution on polarization images, along with polarization-adapted model implementations and pretrained checkpoints.
### Download: [Dataset and Checkpoints (Google Drive)](https://drive.google.com/drive/folders/16z4gJCeky2frSAVkGZwxOEIVvqymqqbD?usp=drive_link)

## Environment Setup
Each model requires a specific environment due to differences in their original implementations. We recommend using the Docker images listed below:

|Model|CUDA|Python|Pytorch|Recommended docker image|
|------|---|---|---|---|
|p-BSRT|11.3|3.8.12|1.11.0|```pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel```|
|p-BurstM|12.1|3.10.14|2.3.0|```pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel```|
|p-Burstormer|11.3|3.8.12|1.11.0|```pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel```|
|p-FBAnet|11.3|3.8.12|1.11.0|```pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel```|
|p-MFIR|11.3|3.8.12|1.11.0|```pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel```|

### Installation
1. **Dependencies**: Install additional packages using the provided files for each models.
   * **Pip**: `pip install -r requirements.txt`
   * **Conda**: `conda env create -f environment.yaml`
   * **Shell script (p-MFIR)**: `./install.sh`
2. **DCNv2 (For p-BSRT)**
```
$ cd model/DCNv2
$ python3 setup.py build develop
$ cd ../../
```
### Directory Structure
Please download and extract the checkpoints and dataset to the appropriate locations. You can modify file paths in ```local.py```.
```
PolarNS/
├── checkpoints/
├── codes/
│   ├── matlab/
│   ├── p-bsrt/
│   ├── p-burstM/
│   ├── p-burstormer/
│   ├── p-fbanet/
│   └── p-mfir/
└── PolarBurstSR/
    ├── test/
    ├── train/
    └── val/
```
**Note on Synthetic Data**: For the synthetic dataset, we utilize the [Sony RSP dataset](https://github.com/sony/polar-densification). Please refer to their official guidelines to obtain the data.
## Test using pretrained model

```
$ cp local_{model name}_{syn or real}.py local.py
$ python test.py
```

## Evaluation
```
$ ./compute_metric.sh # for synthetic dataset using general metric (PSNR, SSIM, LPIPS).
$ ./align_metric.sh # for real dataset using Aligned metric
```
Or you can execute directly for your custom pathes.
```
$ CUDA_VISIBLE_DEVICES=0 python compute_metric.py --gt {path for test set of synthetic dataset} --render {path for rendered image} # for synthetic dataset using general metric (PSNR, SSIM, LPIPS).
$ CUDA_VISIBLE_DEVICES=0 python align_metric.py --gt {path for test set of real dataset} --render {path for rendered image} # for real dataset using Aligned metric
```
## Training
```
$ cp local_{model name}_{syn or real}.py local.py
$ python train.sh -g 0 1 2 3 -p 16010 # python train.sh -g {list of indices for multiple GPUs} -p {port number for DDP} 
$ python train_real.sh -g 0 1 2 3 -p 16010
```
```train.sh``` and ```train.py``` are training codes for synthetic dataset, and ```train_real.sh``` and ```train_real.py``` are finetuning codes for real dataset using aligned metric. You can modify the configuration in `local.py` to customize the settings.
# Citation
```
@InProceedings{Hwang_2025_ICCV,
   author = {Inseung Hwang and Kiseok Choi and Hyunho Ha and Min H. Kim},
   title = {Benchmarking Burst Super-Resolution for Polarization Images:
           Noise Dataset and Analysis},
   booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
   month = {October},
   year = {2025}
} 
```
# Acknowledgement
Our polarization-adapted burst SR models are built upon the following open-source projects:
* [BSRT](https://github.com/Algolzw/BSRT) (MIT License)
* [BurstM](https://github.com/Egkang-Luis/BurstM)
* [Burstormer](https://github.com/akshaydudhane16/Burstormer)
* [FBAnet](https://github.com/yjsunnn/FBANet) (MIT License)
* [MFIR](https://github.com/goutamgmb/deep-rep) (CC BY-NC-SA 4.0)

We thank the authors for their excellent work.

