import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ClassificationEvaluator: 
    
    def __init__(self, num_classes: int, loader: DataLoader, model: nn.Module, device: str):
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise TypeError("num_classes must be a positive integer")
        self.num_classes = num_classes

        if not isinstance(model, nn.Module):
            raise TypeError("model must be a PyTorch nn.Module type")
        self.model = model

        if not isinstance(device, str) or device not in ('cpu', 'cuda'):
            raise ValueError("device must be 'cpu' or 'cuda'")
        self._device = device  # We do not want to change this during runtime
        
        self.loader = loader


    def getDiceScore(self):

        if self.num_classes == 2: 
            num_correct= 0
            num_pixels = 0
            dice_score = 0
            self.model.eval() # Set the model to evaluation mode

            with torch.no_grad(): # To temporarily disable gradient computrations during a specific block of code. 
                            # Gradients are not tracked

                for x,y in self.loader: 
                    x = x.to(self._device)
                    y = y.to(self._device).unsqueeze(1) # unsqueeze used to match the tensor size of x and y
                    preds = torch.sigmoid(self.model(x)) # for binary classification
                    preds = (preds > 0.5).float()
                    num_correct += (preds == y).sum()
                    num_pixels += torch.numel(preds)
                    dice_score += (2 * (preds * y).sum()) / (
                        (preds + y).sum() + 1e-8
                    )

            dice_score = dice_score/len(self.loader)
            self.model.train() # Set model back to training mode

            return dice_score
        else:
            pass # Average Dice score for multiclass classification to be added

    def getDiceScorePerClass(self):
        pass

    def getIOUScore(self):
        pass

    def getAccuracy(self): 
        pass

    def getConfusionMatrix(self):
        pass

    @property
    def device(self):
        return self._device
    

