import yaml
import torch

class ConfigManager:
    def __init__(self, config_file_path):
        with open(config_file_path, 'r') as f: 
            self.config = yaml.safe_load(f)

        if self.config['train']['device'] == 'cuda':  # Confirm if cuda is available incase cuda is selected
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print("==> Using Device : GPU")
        else: 
            self.device ='cpu'
            print("==> Using Device : CPU")

       #model parameters
        self.in_channels = self.config['model']['in_channels']
        self.out_channels = self.config['model']['out_channels']

        #Training hyperparameters
        self.batch_size = self.config['train']['batch_size']
        self.num_epochs = self.config['train']['num_epochs']
        self.learning_rate = float(self.config['train']['learning_rate'])

        #Dataloader parameters
        self.pin_memory = self.config['dataloader']['pin_memory']
        self.num_workers = self.config['dataloader']['num_workers']

        #Dataset Parameters
        self.train_img_dir = self.config['dataset']['train_img_dir']
        self.train_mask_dir = self.config['dataset']['train_mask_dir']
        self.val_img_dir = self.config['dataset']['val_img_dir']
        self.val_mask_dir = self.config['dataset']['val_mask_dir']

        #Training Transforms Parameters (Data Augmentation)
        self.img_height = self.config['train_transform']['resize']['image_height'] 
        self.img_width =  self.config['train_transform']['resize']['image_width']
        self.rotate_limit = self.config['train_transform']['rotate']['limit']
        self.rotate_prob = self.config['train_transform']['rotate']['p']
        self.horizontal_flip_prob = self.config['train_transform']['horizontal_flip']['p']
        self.vertical_flip_prob = self.config['train_transform']['vertical_flip']['p']
        self.normalize_channel_mean = self.config['train_transform']['normalize']['channel_mean']
        self.normalize_channel_std = self.config['train_transform']['normalize']['channel_std']
        self.normalize_max_pixel_value = self.config['train_transform']['normalize']['max_pixel_value']


        self.eval_mode = self.config['eval_mode'] 


            