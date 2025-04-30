import importlib
import os
import torch
from architecture.conv_autoencoder import ConvAutoencoder
from objects.architecture.unet_generator import UNet, ResNetUNet
from objects.architecture.patch_discriminator import PatchDiscriminator


class Stack:
    '''This class holds the data, model, architecture, and results.
    It now also supports GAN models (generator and discriminator).
    '''
    def __init__(self):
        # Original autoencoder model properties
        self.architecture = None
        self.model = None
        
        self.final_model = None
        self.final_history = None
        
        # GAN-specific properties
        self.net_G = None  # Generator network
        self.net_D = None  # Discriminator network
        self.opt_G = None  # Generator optimizer
        self.opt_D = None  # Discriminator optimizer
        self.gan_losses = {}  # Dictionary to track GAN losses
        
        # Data generators
        self.train_generator = None
        self.test_generator = None
        self.val_generator = None
        
        # Image dimensions
        self.img_width = None
        self.img_height = None
        
    def update_datasets(self, train_generator, test_generator, val_generator):
        '''Updates variables dataset_train, test_dataset, val_dataset in stack.
        '''
        self.train_generator = train_generator
        self.test_generator = test_generator
        self.val_generator = val_generator
        
    def create_model(self, hp, model_type='conv_autoencoder'):
        if model_type == "conv_autoencoder":
            self.architecture = ConvAutoencoder()
            self.model = self.architecture.build_model(hp)
        
        return self.model
    
    def create_gan_models(self, input_channels=1, output_channels=2, model_type='unet'):
        """
        Create GAN generator and discriminator models.
        
        Args:
            input_channels (int): Number of input channels (L channel)
            output_channels (int): Number of output channels (ab channels)
            model_type (str): Type of generator model ('unet' or 'resnet_unet')
        
        Returns:
            tuple: (generator, discriminator) networks
        """
        # Create generator based on model type
        if model_type == 'unet':
            self.net_G = UNet(
                input_c=input_channels,
                output_c=output_channels,
                n_down=8,
                num_filters=64
            )
        elif model_type == 'resnet_unet':
            self.net_G = ResNetUNet(
                input_c=input_channels,
                output_c=output_channels,
                pretrained=True
            )
        else:
            raise ValueError(f"Unsupported generator type: {model_type}")
        
        # Create discriminator
        self.net_D = PatchDiscriminator(
            input_c=input_channels + output_channels,  # L + ab channels
            n_down=3,
            num_filters=64
        )
        
        return self.net_G, self.net_D
    
    def finished_model(self, final_model, final_history):
        '''Saves the final model and history for future use.'''
        self.final_model = final_model
        self.final_history = final_history
    
    def finished_gan_training(self, losses):
        """
        Save the GAN training losses.
        
        Args:
            losses (dict): Dictionary of loss histories
        """
        self.gan_losses = losses
        
    def save_gan_models(self, save_dir, model_name):
        """
        Save GAN generator and discriminator models.
        
        Args:
            save_dir (str): Directory to save models
            model_name (str): Base name for the model files
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save generator
        torch.save(self.net_G.state_dict(), 
                  os.path.join(save_dir, f"{model_name}_generator.pth"))
        
        # Save discriminator
        torch.save(self.net_D.state_dict(),
                  os.path.join(save_dir, f"{model_name}_discriminator.pth"))
        
        print(f"Models saved to {save_dir}")
    
    def load_gan_generator(self, model_path, input_channels=1, output_channels=2, model_type='unet'):
        """
        Load a trained generator model.
        
        Args:
            model_path (str): Path to the saved generator model
            input_channels (int): Number of input channels
            output_channels (int): Number of output channels
            model_type (str): Type of generator model ('unet' or 'resnet_unet')
        
        Returns:
            nn.Module: Loaded generator model
        """
        # Create a new model instance
        if model_type == 'unet':
            self.net_G = UNet(
                input_c=input_channels,
                output_c=output_channels
            )
        elif model_type == 'resnet_unet':
            self.net_G = ResNetUNet(
                input_c=input_channels,
                output_c=output_channels
            )
        
        # Load the state dictionary
        self.net_G.load_state_dict(torch.load(model_path))
        self.net_G.eval()  # Set to evaluation mode
        
        return self.net_G
        
    def update_dimensions(self, width, height):
        self.img_width = width
        self.img_height = height

        