# DMGSR
Pytorch implementation of " Learning Distinguishable Degradation Maps for Blind Image Super-Resolution"

## Requirements
- Python 3.9
- PyTorch == 1.13.1
- numpy
- skimage
- imageio
- matplotlib
- cv2
- einops
- basicsr


## Train
replace the file envs/your_envs/lib/python3.9/site-packages/torch/utils/data/dataloader.py with the file data/dataloader.py
### 1. Prepare training data 

1.1 Download the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)  dataset and the [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) dataset.

1.2 Combine the HR images from these two datasets in `your_data_path/DF2K/HR` to build the DF2K dataset. 

### 2. Begin to train
Run `./main.sh` to train on the DF2K dataset. Please update `dir_data` in the bash file as `your_data_path`.


## Test
### 1. Prepare test data 
Download [benchmark datasets](https://github.com/xinntao/BasicSR/blob/a19aac61b277f64be050cef7fe578a121d944a0e/docs/Datasets.md) (e.g., Set5, Set14 and other test sets) and prepare HR/LR images in `your_data_path/benchmark`.


### 2. Begin to test
Run `./test.sh` to test on benchmark datasets. Please update `dir_data` in the bash file as `your_data_path`.


## Quick Test on An LR Image
Run `./quick_test.sh` to test on an LR image. Please update `img_dir` in the bash file as `your_img_path`.


## Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch), [IKC](https://github.com/yuanjunchai/IKC), [DASR](https://github.com/csjliang/DASR) and [MoCo](https://github.com/facebookresearch/moco). We thank the authors for sharing the codes.

