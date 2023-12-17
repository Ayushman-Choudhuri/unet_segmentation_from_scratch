from dataset.carvana import CarvanaDataset
from torch.utils.data import DataLoader

def getCarvanaDataloader(     
    image_dir:str,
    mask_dir:str,
    batch_size:int, 
    transform, 
    num_workers:int  ,
    pin_memory:int
    ): 

    dataset = CarvanaDataset(
                image_dir = image_dir ,
                mask_dir = mask_dir,
                transform =transform,
                )

    dataloader = DataLoader(
                dataset, 
                batch_size = batch_size,
                num_workers = num_workers,
                pin_memory = pin_memory,
                shuffle  = True
                )
    
    return dataloader


