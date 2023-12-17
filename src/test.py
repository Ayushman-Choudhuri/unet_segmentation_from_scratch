import torch 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.dataloaders import getCarvanaDataloader
from utils.logfunctions import loadModel, savePredictions
from utils.evaluators import ClassificationEvaluator

from models.unet import UNet
from utils.config import ConfigLoader

# Load configuration parameters from config file 
config = ConfigLoader('configs/config_carvana.yaml')

def main(): 

    #Setup Test Images Image Augmentations to be applied on test images
    test_transforms = A.Compose(
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


    #Create instance of the unet model 
    model = UNet(in_channels=config.in_channels, out_channels=config.out_channels).to(config.device)

    #Get Test Dataloader
    test_loader = getCarvanaDataloader(config.test_img_dir,
                                    config.test_mask_dir,
                                    config.batch_size,
                                    test_transforms,
                                    config.num_workers,
                                    config.pin_memory) 
    
    loadModel(torch.load("logs/checkpoints/checkpoint_epoch20.pth.tar"), model)
    evaluator = ClassificationEvaluator(2 , test_loader, model, config.device)
    print(f"Dice Score: {evaluator.getDiceScore()}") 
    print(f"IOU Score: {evaluator.getIOUScore()}")
    savePredictions(
            test_loader, model, folder="logs/saved_images/", device=config.device
        )
    

if __name__ == "__main__":
    main()