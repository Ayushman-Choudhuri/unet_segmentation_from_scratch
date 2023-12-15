import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

class CarvanaDataset(Dataset):
    """
    Custom PyTorch dataset class for Carvana Image Masking Challenge Dataset.

    Args:
        image_dir (str): The directory containing input images(Path relative to project root folder)
        mask_dir (str): The directory containing corresponding mask images (Path relative to project root folder)
        transform (callable, optional): A callable transform (Augmentation) from the Albumentations library to be applied
                                        to the images. Default is None.
    """
    def __init__(self, image_dir , mask_dir , transform=None):
        self._image_dir = image_dir
        self._mask_dir = mask_dir
        self._transform = transform
        self._images = os.listdir(image_dir)

    def __len__(self):
        return len(self._images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self._image_dir , self._images[index])
        mask_path = os.path.join(self._mask_dir , self._images[index].replace(".jpg","_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L") , dtype=np.float32) #.convert("L") used to convert to greyscale image
        
        mask[mask==255.0] = 1.0 # So that any values in the mask are either 0 or 1

        if self._transform is not None: #Data augmentation
            augmentations = self._transform(image=image , mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image , mask
    
    @property
    def len(self):
        return len(self._images)
    
    
    

    

    


