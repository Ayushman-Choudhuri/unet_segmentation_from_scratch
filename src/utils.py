import torch 
import torchvision 
from dataset import SegmentationDataset
from torch.utils.data import DataLoader

def saveCheckpoint(state, epoch):
    print(">> Saving Checkpoint")
    filename="checkpoints/"+"checkpoint.pth.tar"+f"_epoch{epoch+1}"
    torch.save(state, filename)

def loadCheckpoint(checkpoint, model):
    print(">> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def getDataloaders(
        
    train_dir,
    train_maskdir,
    val_dir, 
    val_maskdir, 
    batch_size, 
    train_transform, 
    val_transform,
    num_workers  ,
    pin_memory
    ): 

    train_dataset = SegmentationDataset(
                image_dir = train_dir ,
                mask_dir = train_maskdir,
                transform =train_transform,
                )

    train_dataloader = DataLoader(
                train_dataset, 
                batch_size = batch_size,
                num_workers = num_workers,
                pin_memory = pin_memory,
                shuffle  = True
                )
    
    val_dataset = SegmentationDataset(
                image_dir = val_dir ,
                mask_dir = val_maskdir,
                transform =val_transform,
                )
    
    val_dataloader = DataLoader(
                val_dataset,
                batch_size = batch_size,
                num_workers = num_workers,
                pin_memory = pin_memory,
                shuffle = False
                )
    
    return train_dataloader, val_dataloader


def checkAccuracyBC(loader, model, device="cuda"):
    """
    Evaluate the accuracy and dice score of a binary classification model using the provided data loader.

    Parameters:
        loader (torch.utils.data.DataLoader): DataLoader providing batches of input data and ground truth labels.
        model (torch.nn.Module):  Model to be evaluated.
        device (str): Device on which to perform the evaluation (default: "cuda").

    Returns:
        None

    Prints:
        - The number of correct predictions and overall accuracy.
        - The average Dice score across all batches.

    Note:
        - The function assumes that the model is trained for binary classification, and it should be in evaluation mode.
          It iterates through the provided data loader, computes predictions, and evaluates accuracy and Dice score.
        
        -This function goes through the whole dataset (typically validation/test) in batches specified by the 
         BATCH_SIZE parameter. Eg: If my output image size is 160x240 and batch size is 8 and the total images in the 
         dataset is 50, I would get a total number of pixels as 160x240x8x50 = 1920000
          
        - The Dice score is a metric commonly used for evaluating segmentation models.

    Example:
        check_accuracy_binary_classification(val_loader, model, device="cuda")
    """
    
    num_correct= 0
    num_pixels = 0
    dice_score = 0
    model.eval() # Set the model to evaluation mode

    with torch.no_grad(): # To temporarily disable gradient computrations during a specific block of code. 
                          # Gradients are not tracked

        for x,y in loader: 
            x = x.to(device)
            y = y.to(device).unsqueeze(1) # unsqueeze used to match the tensor size of x and y
            preds = torch.sigmoid(model(x)) # for binary classification
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
    acc = (num_correct/num_pixels)*100
    print(
        f"Got {num_correct}/{num_pixels} with accuracy: {acc}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train() # Set model back to training mode

def savePredictions( loader, model, folder="saved_images/", device="cuda") :
    model.eval() # Set model to evaluation mode
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
    model.train() # Set model to training mode