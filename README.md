# UNET Image Segmentation

This project implements a slightly modified UNET architecture and uses it to perform binary segmentation as well as multiclass segmentation. 

Compared to the original model [1], I have added a 2d Batch Normalization layers [2] between every Conv2d and ReLU layer in the original UNET architecture. This would accelerate learning by enabling a smoother optimization landscape making the optimization algorithms converge faster.

## 1. Setup

After cloning the repository , follow the following steps to run project setup. 

**Note**: All commands are to be run from the project root folder. 

### 1.1: Install Dependencies 

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

### 1.2: Folder Structure

Create the following top folder structure

```bash 
unet_segmentation_from_scratch
├── configs
├── datasets
│   └── carvana
├── images
├── logs
│   ├── checkpoints
│   └── saved_images
├── runs
├── src
├── unetseg_venv
├── .gitignore
├── README.md
└── requirements.txt

```
## 2. Datasets

### 2.1: Carvana Dataset

Download the [Carvana Dataset](https://www.kaggle.com/c/carvana-image-masking-challenge/data) from the kaggle page of the Carvana Image Masking Challenge. We need just the data from the folders named train and train_masks. The train folder contains the training set of images and the train_masks contains the .gif mask file for each of the corresponding training images. 

We would be deriving the train , test and validation datasets needed to train , validate and test the Unet model from the train folder of the Carvana Dataset. The train.zip file contains 5088 RGB images with size of 1918 x 1280 pixels.

From the 5088 images, separate out 50 images in val_images and 50 in the test_images and also move the corresponding masks in the test_masks and val_masks folder.

The dataset folder structure should look like this after this:

```bash
datasets
└── carvana
    ├── test_images      # 50 images in .jpg format
    ├── test_masks       # 50 masks in .gif format
    ├── train_images     # 4988 images in .jpg format
    ├── train_masks      # 4988 masks in .gif format
    ├── val_images       # 50 images in .jpg format
    └── val_masks        # 50 masks in .gif format
```

## 5. Running the Project 

### 5.1 Setting up dataset configurations

In the config_carvana.yaml file, tune the training hyperparameters to adjust for your training hardware setup

### 5.2 Running training loop: 

```bash

python3 src/train.py

```

## 6. Results 



## 7. References 

[1] [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

[2] [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
