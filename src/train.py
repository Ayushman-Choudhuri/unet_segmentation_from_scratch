import torch 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm 
import torch.nn as nn
import torch.optim as optim 
from model import UNet
from configmanager import ParameterManager


#Import Utility Functions 
from utils import (
    loadCheckpoint,
    saveCheckpoint,
    getDataloaders,
    checkAccuracyBC,
    savePredictions,
)

# Load Parameters from config file 
param = ParameterManager('configs/config.yaml')


def train_step( loader , model, optimizer , loss_fn, scaler, epoch):
    loop = tqdm(loader)

    for batch_idx , (input_data , target_labels) in enumerate(loop):
        input_data = input_data.to(device= param.device)
        target_labels = target_labels.float().unsqueeze(1).to(device=param.device)  # to match the tensor shape of input data

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
            A.Resize(height=param.img_height, width=param.img_width),
            A.Rotate(limit=param.rotate_limit, p=param.rotate_prob),
            A.HorizontalFlip(p= param.horizontal_flip_prob),
            A.VerticalFlip(p=param.vertical_flip_prob),
            A.Normalize(
                mean=[param.normalize_channel_mean, param.normalize_channel_mean, param.normalize_channel_mean],
                std=[param.normalize_channel_std, param.normalize_channel_std, param.normalize_channel_std],
                max_pixel_value=param.normalize_max_pixel_value,
            ),
            ToTensorV2(),
        ],
    )

    #Setup image augmentations on validation data
    val_transforms = A.Compose(
        [
            A.Resize(height=param.img_height, width=param.img_width),
            A.Normalize(
                mean=[param.normalize_channel_mean, param.normalize_channel_mean, param.normalize_channel_mean],
                std=[param.normalize_channel_std, param.normalize_channel_std, param.normalize_channel_std],
                max_pixel_value=param.normalize_max_pixel_value,
            ),
            ToTensorV2(),
        ],
    )
    
    # Create instance of UNET model class 
    model = UNet(in_channels=param.in_channels, out_channels=param.out_channels).to(param.device) 
    
    #Setup Loss Function based on number of output classes
    if param.out_channels == 1: 
        loss_fn = nn.BCEWithLogitsLoss() #  Here we are going with BCE(Binary Cross Entropy) with logits loss as we are doing binary classification of pixels. 
                                     #  Also nn.BCEWithLogitsLoss is more stable than nn.BCEloss
    else: 
        loss_fn = nn.CrossEntropyLoss() # You can shift to CrossEntropy loss if you want multiclass segmentation.


    # Setup Optimizer
    optimizer = optim.Adam(model.parameters(), lr=param.learning_rate) # Setup ADAM optimizer

    # Setup Dataloaders
    train_loader, val_loader = getDataloaders(
        param.train_img_dir,
        param.train_mask_dir,
        param.val_img_dir,
        param.val_mask_dir,
        param.batch_size,
        train_transform,
        val_transforms,
        param.num_workers,
        param.pin_memory
    )
    

    if param.eval_mode: 
        loadCheckpoint(torch.load("checkpoints/checkpoint.pth.tar_epoch16"), model)
        checkAccuracyBC(val_loader,model, device=param.device)
        savePredictions(
            val_loader, model, folder="saved_images/", device=param.device
        )

    else: 
        
        #Setup Scaler to optimize compute efficiency in training loops by dynamically adjusting the scale of the gradient during backward pass
        # This is done to avoid the problem of gradient overflow or underflow.
        scaler = torch.cuda.amp.GradScaler()

        # Empty the GPU cache before training starts
        torch.cuda.empty_cache()
        
        for epoch in range(param.num_epochs):

            print(f"EPOCH {epoch+1}")

            train_step(train_loader, model, optimizer, loss_fn, scaler,epoch)

            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }

            saveCheckpoint(checkpoint,epoch)

            # check accuracy
            checkAccuracyBC(val_loader, model, device=param.device)

        # Close the SummaryWriter
        #writer.close()

if __name__ == "__main__":
    main()

