import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ClassificationEvaluator: 
    
    def __init__(self , num_classes:int , loader:DataLoader , model:nn.Module , device:str):
        
        self.num_classes = num_classes

        if not isinstance(num_classes , int ):
            raise TypeError("num_classes must be a integer > 0")

        self.model = model

        if not isinstance(model, nn.Module):
            raise TypeError("model must be pytorch nn.Module type")


        self._device = device #We do not want to change this during runtime

        
    def getDiceScoreAvg(self):
        pass

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
    

