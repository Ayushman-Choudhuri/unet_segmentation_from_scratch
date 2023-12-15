from dataset.carvana import CarvanaDataset
from torch.utils.data import DataLoader


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

    train_dataset = CarvanaDataset(
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
    
    val_dataset = CarvanaDataset(
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
    
    #print(train_dataset.len)
    
    return train_dataloader, val_dataloader