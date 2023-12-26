# UNET Image Segmentation

This project implements a slightly modified UNET architecture and uses it to perform binary segmentation as well as multiclass segmentation. 

The Image segmentation has been performed on two datasets. The carvana dataset and the cityscapes dataset. The carvana dataset was initially used to test the perfomance of the UNET model for a simple binary segmentation task. After that the UNET model has been applied for multiclass segmentation on the cityscapes dataset.


**Changes in UNET:**

Compared to the original model [1], I have added a 2d Batch Normalization layers [2] between every Conv2d and ReLU layer in the original UNET architecture. This would accelerate learning by enabling a smoother optimization landscape making the optimization algorithms converge faster.

## 1. Branches

**main:** The main branch holds the implementation of the binary segmentation on the carvana dataset.

**cityscapes:** The cityscapes branch holds the implementation of the multiclass segmentation using the cityscapes dataset


## 2. Setup

After cloning the repository , follow the following steps to run project setup. 

**Note**: All commands are to be run from the project root folder. 

### 2.1: Install Dependencies 

**Step1**: Create a python virtual environment and activate it

```bash
python3 -m venv unetseg_venv && source unetseg_venv/bin/activate
```

**Step2**: Install dependencies from requirements.txt

```bash 
pip3 install -r requirements.txt
```
**Step3**: Install Pytorch based on your system configs [Link](https://pytorch.org/): 

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2.2: Folder Structure


## 3. Datasets

## 4. Running the Project 

## 5. Results 



## 6. References 

[1] [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

[2] [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

[3] [CityscpapesScripts GitHub](https://github.com/mcordts/cityscapesScripts/tree/master)
