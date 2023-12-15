import torch 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm 
import torch.nn as nn
import torch.optim as optim 
from models.unet import UNet
from utils.config import ConfigManager
from utils.dataloaders import getDataloaders
from utils.evaluators import ClassificationEvaluator
from utils.logfunctions import loadCheckpoint, saveCheckpoint , savePredictions


# Load configuration parameters from config file 
config = ConfigManager('configs/config_carvana.yaml')

def trainStep( loader , model, optimizer , loss_fn, scaler, epoch):
    loop = tqdm(loader)

    for batch_idx , (input_data , target_labels) in enumerate(loop):
        input_data = input_data.to(device= config.device)
        target_labels = target_labels.float().unsqueeze(1).to(device=config.device)  # to match the tensor shape of input data

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

    return loss


def main(): 

    # Setup Image augmentations on training data
    train_transforms = A.Compose(
        [
            A.Resize(height=config.img_height, width=config.img_width),
            A.Rotate(limit=config.rotate_limit, p=config.rotate_prob),
            A.HorizontalFlip(p= config.horizontal_flip_prob),
            A.VerticalFlip(p=config.vertical_flip_prob),
            A.Normalize(
                mean=[config.normalize_channel_mean, config.normalize_channel_mean, config.normalize_channel_mean],
                std=[config.normalize_channel_std, config.normalize_channel_std, config.normalize_channel_std],
                max_pixel_value=config.normalize_max_pixel_value,
            ),
            ToTensorV2(),
        ],
    )

    #Setup image augmentations on validation data
    val_transforms = A.Compose(
        [
            A.Resize(height=config.img_height, width=config.img_width),
            A.Normalize(
                mean=[config.normalize_channel_mean, config.normalize_channel_mean, config.normalize_channel_mean],
                std=[config.normalize_channel_std, config.normalize_channel_std, config.normalize_channel_std],
                max_pixel_value=config.normalize_max_pixel_value,
            ),
            ToTensorV2(),
        ],
    )
    
    # Create instance of unet model class 
    model = UNet(in_channels=config.in_channels, out_channels=config.out_channels).to(config.device) 
    
    #Setup Loss Function based on number of output classes
    if config.out_channels == 1: 
        loss_fn = nn.BCEWithLogitsLoss() #  Here we are going with BCE(Binary Cross Entropy) with logits loss as we are doing binary classification of pixels. 
                                     #  Also nn.BCEWithLogitsLoss is more stable than nn.BCEloss
    else: 
        loss_fn = nn.CrossEntropyLoss() # You can shift to CrossEntropy loss if you want multiclass segmentation.


    # Setup Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate) # Setup ADAM optimizer


    # Setup Dataloaders from training and Validation datasets
    train_loader, val_loader = getDataloaders(  config.train_img_dir,
                                                config.train_mask_dir,
                                                config.val_img_dir,
                                                config.val_mask_dir,
                                                config.batch_size,
                                                train_transforms,
                                                val_transforms,
                                                config.num_workers,
                                                config.pin_memory
                                             )
    

    if config.eval_mode: 
        loadCheckpoint(torch.load("checkpoints/checkpoint.pth.tar_epoch16"), model)
        evaluator = ClassificationEvaluator(2 , val_loader, model, config.device)
        print(f"Dice Score: {evaluator.getDiceScore()}")
        
        savePredictions(
            val_loader, model, folder="saved_images/", device=config.device
        )

    else: 
        
        #Setup Scaler to optimize compute efficiency in training loops by dynamically adjusting the scale of the gradient during backward pass
        # This is done to avoid the problem of gradient overflow or underflow.
        scaler = torch.cuda.amp.GradScaler()

        # Empty the GPU cache before training starts
        torch.cuda.empty_cache()
        
        for epoch in range(config.num_epochs):

            print(f"EPOCH {epoch+1}")

            epoch_loss = trainStep(train_loader, model, optimizer, loss_fn, scaler,epoch)

            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }

            saveCheckpoint(checkpoint,epoch)

            # check accuracy
            evaluator = ClassificationEvaluator(2 , val_loader, model, config.device)
            print(f"Dice Score: {evaluator.getDiceScore():.4f}")
            print(f"Epoch Loss: {epoch_loss:.4f}")


        # Close the SummaryWriter
        #writer.close()

if __name__ == "__main__":
    main()

