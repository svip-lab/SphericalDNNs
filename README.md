# SphericalDNNs

### Introduction
This repo contains the codes that used in paper *Saliency Detection in 360° Videos*.

### Requirements
  - Python 3.6 is required.
  - Module Dependencies:
    - 0.3.0 <= torch <= 0.4.1 
    - numpy >= 1.13

### File structure
```
- train_lstm.py & train_unet.py
  purpose: Provides a simple training codes for Sphercial lstm / Spherical U-Net model that uses spherical convolution.
- datasets/
  purpose: Provides the dataloader for our dataset to train models.
- models/baseline_unet.py & baseline_convlstm.py
  purpose: Provides the implementation of Spherical U-Net & Sphercial lstm.
- tools/baselines/baseline_unet.sh & baseline_convlstm.sh
  purpose: Provides the cmd to run the training codes. 
- opts.py & ref.py
  purpose: Provides some used parameters.
- sconv
  - functional
    - common.py
      Purpose: Contains some helper functions used in sphercal convolution.
    - sconv.py
      Purpose: Provides the spherical convolution function for Pytorch.
    - spad.py
      Purpose: Provides the spherical pooling function for Pytorch.
  - module
    - sconv.py
      Purpose: Provides the spherical convolution module for Pytorch.
    - smse.py
      Purpose: Provides the spherical mean-square loss module for Pytorch.
    - spad.py
      Purpose: Provides the spherical padding module for Pytorch.
    - spool.py
      Purpose: Provides the spherical pooling module for Pytorch.
```

### Usage
  The spherical convolution is written in pure python with pytorch, so that no compiling proceedure is needed. One can just pull and run all the codes in this repo. We currently provide a sample model in `test.py` that uses spherical convolution layers. The model and checkpoint that used in original paper will be released later.
  
### Known issues
  - The process of determining the kernel area at different θ locations is unstable, which will cause output feature maps contain some `nan` values. However, this bug seems to have minor effects during training and testing. We will try to fix it later.
  
### Dataset
  You can download our dataset [[Baidu Pan]](https://pan.baidu.com/s/18equcFntAomwEEP3TgHhFw) (code:p0a5), which consists of 104 videos. There are 12 zip files, and train/test index. After downloading these zips, unzip them together. There are 104 files and 'vinfo.pkl'. Each file consists of *.jpg (RGB image) and *.npy (ground truth heatmaps). The 'pkl' file consists of the original groud truth gaze points of the observers.    


### TODO
  - [x] Release core functions and modules
  - [x] Release training code for saliency detection
  - [ ] Resolve the math unstability when calculating the kernel area at different θ locations
  - [ ] Rewrite spherical convolution for torch 0.4+

### License

This project is released under the MIT License (refer to the LICENSE file for details).

### Citing

If you find this repo useful in your research, please consider citing:
```
    @InProceedings{Zhang_2018_ECCV,
        author = {Zhang, Ziheng and Xu, Yanyu and Yu, Jingyi and Gao, Shenghua},
        title = {Saliency Detection in 360° Videos},
        booktitle = {The European Conference on Computer Vision (ECCV)},
        month = {September},
        year = {2018}
    }
    
    @article{xu2022spherical,
      title={Spherical DNNs and Their Applications in 360$\^{}$\backslash$circ $∘ Images and Videos},
      author={Xu, Yanyu and Zhang, Ziheng and Gao, Shenghua},
      journal={IEEE Transactions on Pattern Analysis \& Machine Intelligence},
      volume={44},
      number={10},
      pages={7235--7252},
      year={2022},
      publisher={IEEE Computer Society}
    }
```
