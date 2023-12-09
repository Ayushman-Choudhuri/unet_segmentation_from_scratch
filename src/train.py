import torch 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm 
import torch.nn as nn
import torch.optim as optim 
from model import UNET

#Hyperparameters

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 100
NUM_WORKERS = 2
LEARNING_RATE = 1e-4
IMAGE_HEIGHT = 160 
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL= False
TRAIN_IMG_DIR = "dataset/train_images/"
TRAIN_MASK_DIR = "dataset/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"


def train( loader , model, optimizer , loss_fn, scaler):
    pass

def main(): 
    pass

if __name__ == "__main__":
    main()

