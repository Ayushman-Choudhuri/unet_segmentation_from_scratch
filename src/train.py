import torch 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm 
import torch.nn as nn
import torch.optim as optim 
from model import UNET

#Import Utility Functions 

from utils import (
    load_checkpoint,
    save_checkpoint,
    get_dataloaders,
    check_accuracy_binary_classification,
    save_predictions_as_imgs,
)

#Hyperparameters

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 50
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
IMAGE_HEIGHT = 160 
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL= False
TRAIN_IMG_DIR = "../dataset/train_images/"
TRAIN_MASK_DIR = "../dataset/train_masks/"
VAL_IMG_DIR = "../dataset/val_images/"
VAL_MASK_DIR = "../dataset/val_masks/"


def train( loader , model, optimizer , loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx , (input_data , target_labels) in enumerate(loop):
        input_data = input_data.to(device= DEVICE)
        target_labels = target_labels.float().unsqueeze(1).to(device=DEVICE)  # to match the tensor shape of input data

        #Forward Pass
        with torch.cuda.amp.autocast(): # To enable Automatic Mixed Precision (amp) feature 
            predictions=model(input_data)
            loss = loss_fn(predictions, target_labels)

        
        # Backpropagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())  # adds additional loss stats to display at the end of the tqdm bar

        # Empty the GPU cache after each epoch
        torch.cuda.empty_cache()


def main(): 

    # Setup Image augmentations on training data
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )


    #Setup image augmentations on validation data

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )



    
    model = UNET(in_channels=3, out_channels=1).to(DEVICE) # Create instance of UNET model class 
    loss_fn = nn.BCEWithLogitsLoss() # Define loss function. Here we are going with BCE(Binary Cross Entropy) with logits loss as we are doing binary classification of pixels. 
                                     # You can shift to Crossentropy loss if you want multiclass segmentation. Also nn.BCEWithLogitsLoss is more stable than nn.BCEloss
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Setup ADAM optimizer

    train_loader, val_loader = get_dataloaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("../checkpoints/my_checkpoint.pth.tar"), model)


    check_accuracy_binary_classification(val_loader, model, device=DEVICE)

    scaler = torch.cuda.amp.GradScaler()

    # Empty the GPU cache before training starts
    torch.cuda.empty_cache()
    
    for epoch in range(NUM_EPOCHS):
        train(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy_binary_classification(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="../saved_images/", device=DEVICE
        )

if __name__ == "__main__":
    main()

