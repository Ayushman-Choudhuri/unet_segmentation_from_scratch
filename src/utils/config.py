import yaml
import torch

class ConfigLoader:
    def __init__(self, config_file_path):
        try:
            with open(config_file_path, 'r') as f: 
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at: {config_file_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error loading YAML from {config_file_path}: {e}")

        if self.config['train']['device'] == 'cuda':  # Confirm if cuda is available incase cuda is selected
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print("==> Using Device : GPU")
        else: 
            self.device ='cpu'
            print("==> Using Device : CPU")

       #model parameters
        self.in_channels = int(self.config['model']['in_channels'])
        self.out_channels = int(self.config['model']['out_channels'])

        #Training hyperparameters
        self.batch_size = int(self.config['train']['batch_size'])
        self.num_epochs = int(self.config['train']['num_epochs'])
        self.learning_rate = float(self.config['train']['learning_rate'])

        #Dataloader parameters
        self.pin_memory = bool(self.config['dataloader']['pin_memory'])
        self.num_workers = int(self.config['dataloader']['num_workers'])

        #Dataset directories
        self.train_img_dir = str(self.config['dataset']['train_img_dir'])
        self.train_mask_dir = str(self.config['dataset']['train_mask_dir'])
        self.val_img_dir = str(self.config['dataset']['val_img_dir'])
        self.val_mask_dir = str(self.config['dataset']['val_mask_dir'])
        self.test_img_dir = str(self.config['dataset']['test_img_dir'])
        self.test_mask_dir = str(self.config['dataset']['test_mask_dir'])

        #Log directories
        self.checkpoint_dir = str((self.config['logs']['checkpoints']['checkpoint_dir']))

        #Transforms Parameters (Data Augmentation)
        self.img_height = int(self.config['transform']['resize']['image_height'])
        self.img_width =  int(self.config['transform']['resize']['image_width'])
        self.rotate_limit = int(self.config['transform']['rotate']['limit'])
        
        self.rotate_prob = float(self.config['transform']['rotate']['p'])
        self.horizontal_flip_prob = float(self.config['transform']['horizontal_flip']['p'])
        self.vertical_flip_prob = float(self.config['transform']['vertical_flip']['p'])
        self.normalize_channel_mean = float(self.config['transform']['normalize']['channel_mean'])
        self.normalize_channel_std = float(self.config['transform']['normalize']['channel_std'])
        self.normalize_max_pixel_value = float(self.config['transform']['normalize']['max_pixel_value'])



            