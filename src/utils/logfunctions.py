import torch
import torchvision

def saveCheckpoint(state, epoch):
    print(">> Saving Checkpoint")
    filename="logs/checkpoints/"+"checkpoint.pth.tar"+f"_epoch{epoch+1}"
    torch.save(state, filename)

def loadCheckpoint(checkpoint, model):
    print(">> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def savePredictions( loader, model, folder="logs/saved_images/", device="cuda") :
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