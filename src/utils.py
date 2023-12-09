import torch 
import torchvision 
from dataset import SegmentationDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print(">> Saving Checkpoint")
    torch.save(state, filename)

def load_checkpoint (checkpoint, model):
    print(">> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_dataloaders(
        
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


def check_accuracy_binary_classification(loader, model, device="cuda"):
    num_correct= 0
    num_pixels = 0
    dice_score = 0
    model.eval() # Set the model to evaluation mode

    with torch.no_grad(): # To temporarily disable gradient computrations during a specific block of code. 
                          # Gradients are not tracked

        for x,y in  loader: 
            x = x.to(device)
            y = y.to(device).unsqueeze(1) # unsqueeze used to match the tensor size of x and y
            preds = torch.sigmoid(model(x)) # for binary classification
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train() # Set model back to training mode

def save_predictions_as_imgs( loader, model, folder="saved_images/", device="cuda") :
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