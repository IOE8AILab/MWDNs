# MWDNs
This repository contains PyTorch implementation for the paper: "MWDNs: reconstruction in multi-scale feature spaces for lensless imaging". It is an efficient image reconstruction algorithm that combines Wiener filtering and deep learning for lensless cameras based on pseudo random phase modulation.
### Setup:
Clone this project using:
```
git clone https://github.com/IOE8AILab/MWDNs.git
```
The code is developed using Python 3.9, PyTorch 1.9.0. The GPU we used is NVIDIA RTX 3090. 

### Dataset 
Download [dateset](https://drive.google.com/file/d/1wP3CWahU8Mxqod2F-VcUvnGz33XB9wwF/view?usp=share_link). It contains three files ('blur', 'gt', 'codes'). In the 'blur', there are 25,000 blurry images captured on the display by the lensless camera, while in the 'gt', it is ground truth corresponding to the 'blur'. The 'codes' includes code, PSF, and lensless images for real objects.

### Training and Testing
To preprocess dataset and train the entire framework:
```
cd MWDNs/
python h5_datamake.py
python train_model.py
python test.py
```
Please make sure your path is set properly for the dataset.

### Contact Us
In case of any queries, please reach out to [liying](mailto:liying192@mails.ucas.ac.cn?subject=[MWDNs%20Code%20Query]).
