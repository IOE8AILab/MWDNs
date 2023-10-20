# MWDNs
This repository contains PyTorch implementation for the paper: "MWDNs: reconstruction in multi-scale feature spaces for lensless imaging". It is an efficient image reconstruction algorithm that combines Wiener filtering and deep learning for lensless cameras based on pseudo random phase modulation.
### Setup:
Clone this project using:
```
git clone https://github.com/IOE8AILab/MWDNs.git
```
The code is developed using Python 3.9, PyTorch 1.9.0. The GPU we used is NVIDIA RTX 3090. 

### Dataset 
Download [dateset](https://drive.google.com/file/d/1wP3CWahU8Mxqod2F-VcUvnGz33XB9wwF/view?usp=share_link)
You should then have the following directory structure:
```bash
.
|-- blur
|-- codes
|   |-- blur_capture_real_object
|   `-- psf
|-- gt
```

### Training
To preprocess dataset and train the entire framework:
```
cd SH-CNN/
python DataMake.py
python Train_Model.py
```

### Accelerating
To accelerate the model by tensorrt:
```
python cnn2trt.py
```

Please make sure your path is set properly for the dataset.
