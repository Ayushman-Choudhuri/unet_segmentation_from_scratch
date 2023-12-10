import torch 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm 
import torch.nn as nn
import torch.optim as optim 
from model import UNET
import yaml

#Import Utility Functions 
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_dataloaders,
    check_accuracy_binary_classification,
    save_predictions_as_imgs,
)

# Load Parameters

with open('configs/config.yaml' , 'r') as f: 
    config=yaml.safe_load(f)

if config['train']['device'] == 'cuda':  # Confirm if cuda is available incase cuda is selected
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
else: 
    DEVICE ='cpu'

print(f"==> Using Device :{DEVICE}")

#model parameters
IN_CHANNELS = config['model']['in_channels']
OUT_CHANNELS = config['model']['out_channels']

#Training hyperparameters
BATCH_SIZE = config['train']['batch_size']
NUM_EPOCHS = config['train']['num_epochs']
LEARNING_RATE = float(config['train']['learning_rate'])

#Dataloader parameters
PIN_MEMORY = config['dataloader']['pin_memory']
NUM_WORKERS = config['dataloader']['num_workers']

#Dataset Parameters
TRAIN_IMG_DIR = config['dataset']['train_img_dir']
TRAIN_MASK_DIR = config['dataset']['train_mask_dir']
VAL_IMG_DIR = config['dataset']['val_img_dir']
VAL_MASK_DIR = config['dataset']['val_mask_dir']

#Training Transforms Parameters (Data Augmentation)
IMAGE_HEIGHT = config['train_transform']['resize']['image_height'] 
IMAGE_WIDTH =  config['train_transform']['resize']['image_width']
ROTATE_LIMIT = config['train_transform']['rotate']['limit']
ROTATE_PROB = config['train_transform']['rotate']['p']
HORIZONTAL_FLIP_PROB = config['train_transform']['horizontal_flip']['p']
VERTICAL_FLIP_PROB = config['train_transform']['vertical_flip']['p']
NORMALIZE_CHANNEL_MEAN = config['train_transform']['normalize']['channel_mean']
NORMALIZE_CHANNEL_STD = config['train_transform']['normalize']['channel_std']
NORMALIZE_MAX_PIX_VALUE = config['train_transform']['normalize']['max_pixel_value']

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
            A.Rotate(limit=ROTATE_LIMIT, p=ROTATE_PROB),
            A.HorizontalFlip(p=HORIZONTAL_FLIP_PROB),
            A.VerticalFlip(p=VERTICAL_FLIP_PROB),
            A.Normalize(
                mean=[NORMALIZE_CHANNEL_MEAN, NORMALIZE_CHANNEL_MEAN, NORMALIZE_CHANNEL_MEAN],
                std=[NORMALIZE_CHANNEL_STD, NORMALIZE_CHANNEL_STD, NORMALIZE_CHANNEL_STD],
                max_pixel_value=NORMALIZE_MAX_PIX_VALUE,
            ),
            ToTensorV2(),
        ],
    )

    #Setup image augmentations on validation data
    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[NORMALIZE_CHANNEL_MEAN, NORMALIZE_CHANNEL_MEAN, NORMALIZE_CHANNEL_MEAN],
                std=[NORMALIZE_CHANNEL_STD, NORMALIZE_CHANNEL_STD, NORMALIZE_CHANNEL_STD],
                max_pixel_value=NORMALIZE_MAX_PIX_VALUE,
            ),
            ToTensorV2(),
        ],
    )
    
    # Create instance of UNET model class 
    model = UNET(in_channels=3, out_channels=1).to(DEVICE) 
    
    #Setup Loss Function
    if OUT_CHANNELS == 1: 
        loss_fn = nn.BCEWithLogitsLoss() #  Here we are going with BCE(Binary Cross Entropy) with logits loss as we are doing binary classification of pixels. 
                                     #  Also nn.BCEWithLogitsLoss is more stable than nn.BCEloss
    else: 
        loss_fn = nn.CrossEntropyLoss() # You can shift to CrossEntropy loss if you want multiclass segmentation.


    # Setup Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Setup ADAM optimizer

    # Setup Dataloaders
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
    
    #Setup Scaler to optimize compute efficiency in training loops by dynamically adjusting the scale of the gradient during backward pass
    # This is done to avoid the problem of gradient overflow or underflow.
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

        # # print some examples to a folder
        # save_predictions_as_imgs(
        #     val_loader, model, folder="../saved_images/", device=DEVICE
        # )

if __name__ == "__main__":
    main()

