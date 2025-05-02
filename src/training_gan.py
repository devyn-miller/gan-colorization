import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import time
from tqdm import tqdm
from skimage.color import rgb2lab
from datetime import datetime

# Import local modules
import importlib
import objects.stack as stack_module
importlib.reload(stack_module)
from objects.stack import Stack
from gan_utils import (
    GANLoss, AverageMeter, create_loss_meters, 
    update_losses, lab_to_rgb, visualize, log_results, init_weights
)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class LABDataset(Dataset):
    """
    Dataset for LAB color space images for colorization task
    """
    def __init__(self, rgb_images):
        """
        Initialize dataset with RGB images

        Args:
            rgb_images: RGB images from the data generator
        """
        self.rgb_images = rgb_images
        
    def __len__(self):
        return len(self.rgb_images)
    
    def __getitem__(self, idx):
        # Convert RGB to LAB color space
        img_rgb = self.rgb_images[idx]
        img_lab = rgb2lab(img_rgb).astype("float32")
        
        # Extract L and ab channels and normalize
        L = torch.from_numpy(img_lab[:, :, 0:1]).permute(2, 0, 1) / 50. - 1.  # Scale to [-1, 1]
        ab = torch.from_numpy(img_lab[:, :, 1:3]).permute(2, 0, 1) / 110.  # Scale to [-1, 1]
        
        return {'L': L, 'ab': ab}

def create_dataloaders(stack, batch_size=16):
    """
    Create PyTorch dataloaders from stack generators
    
    Args:
        stack (Stack): Stack object with data generators
        batch_size (int): Batch size for dataloaders
        
    Returns:
        tuple: (train_dl, val_dl, test_dl) dataloaders
    """
    # Extract entire data from generators to convert to PyTorch datasets
    train_images = []
    for batch in stack.train_generator:
        train_images.extend(batch)
        # Limit the amount of data if it's too large
        if len(train_images) > 10000:  # Adjust as needed
            break
    
    val_images = []
    for batch in stack.val_generator:
        val_images.extend(batch)
        if len(val_images) > 2000:  # Adjust as needed
            break
    
    test_images = []
    for batch in stack.test_generator:
        test_images.extend(batch)
        if len(test_images) > 2000:  # Adjust as needed
            break
    
    # Create PyTorch datasets
    train_ds = LABDataset(train_images)
    val_ds = LABDataset(val_images)
    test_ds = LABDataset(test_images)
    
    # Create dataloaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    
    print(f"Created dataloaders - Train: {len(train_dl)}, Val: {len(val_dl)}, Test: {len(test_dl)}")
    
    return train_dl, val_dl, test_dl

class MainModel(nn.Module):
    """
    Main GAN model that contains both generator and discriminator
    """
    def __init__(self, net_G=None, net_D=None, lr_G=2e-4, lr_D=2e-4, 
                 beta1=0.5, beta2=0.999, lambda_L1=100.):
        """
        Initialize the main model
        
        Args:
            net_G (nn.Module): Generator network
            net_D (nn.Module): Discriminator network
            lr_G (float): Learning rate for generator
            lr_D (float): Learning rate for discriminator
            beta1 (float): Beta1 parameter for Adam optimizer
            beta2 (float): Beta2 parameter for Adam optimizer
            lambda_L1 (float): Weight for L1 loss
        """
        super().__init__()
        
        self.device = device
        self.lambda_L1 = lambda_L1
        
        # Initialize networks
        if net_G is None:
            raise ValueError("Generator network must be provided")
        else:
            self.net_G = net_G.to(self.device)
            
        if net_D is None:
            raise ValueError("Discriminator network must be provided")
        else:
            self.net_D = net_D.to(self.device)
            
        # Initialize networks with normal distribution weights
        self.net_G = init_weights(self.net_G)
        self.net_D = init_weights(self.net_D)
        
        # Set loss functions
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        
        # Set optimizers
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))
    
    def set_requires_grad(self, model, requires_grad=True):
        """
        Set requires_grad=True or False for all parameters in the model
        
        Args:
            model (nn.Module): Model to update
            requires_grad (bool): Whether gradients are required
        """
        for p in model.parameters():
            p.requires_grad = requires_grad
        
    def setup_input(self, data):
        """
        Set up input data
        
        Args:
            data (dict): Input data with 'L' and 'ab' keys
        """
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)
        
    def forward(self):
        """
        Forward pass - generate fake color
        """
        self.fake_color = self.net_G(self.L)
    
    def backward_D(self):
        """
        Calculate GAN loss for the discriminator and backpropagate
        """
        # Fake; stop backprop to the generator by detaching fake_color
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        
        # Real
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        
        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
    
    def backward_G(self):
        """
        Calculate losses for the generator and backpropagate
        """
        # First, G(A) should fake the discriminator
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        
        # Second, G(A) = B
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        
        # Combined loss
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
    
    def optimize(self):
        """
        Update weights for both generator and discriminator
        """
        # Forward pass
        self.forward()
        
        # Update D
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()
        
        # Update G
        self.net_G.train()
        self.set_requires_grad(self.net_D, False) # Don't need to calculate gradients for D when updating G
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()

def pretrain_generator(net_G, train_dl, val_dl=None, epochs=10, lr=1e-4):
    """
    Pretrain the generator with L1 loss only
    
    Args:
        net_G (nn.Module): Generator network
        train_dl (DataLoader): Training dataloader
        val_dl (DataLoader): Validation dataloader
        epochs (int): Number of training epochs
        lr (float): Learning rate
        
    Returns:
        net_G (nn.Module): Pretrained generator
    """
    print(f"Pretraining generator for {epochs} epochs...")
    optimizer = optim.Adam(net_G.parameters(), lr=lr)
    criterion = nn.L1Loss()
    net_G = net_G.to(device)
    
    for e in range(epochs):
        # Training
        net_G.train()
        train_loss = AverageMeter()
        
        for data in tqdm(train_dl, desc=f"Epoch {e+1}/{epochs} Training"):
            L, ab = data['L'].to(device), data['ab'].to(device)
            optimizer.zero_grad()
            
            # Forward pass
            preds = net_G(L)
            loss = criterion(preds, ab)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss.update(loss.item(), L.size(0))
        
        # Validation if provided
        if val_dl is not None:
            net_G.eval()
            val_loss = AverageMeter()
            
            with torch.no_grad():
                for data in tqdm(val_dl, desc=f"Epoch {e+1}/{epochs} Validation"):
                    L, ab = data['L'].to(device), data['ab'].to(device)
                    preds = net_G(L)
                    loss = criterion(preds, ab)
                    val_loss.update(loss.item(), L.size(0))
            
            print(f"Epoch {e+1}/{epochs}: Train L1 Loss: {train_loss.avg:.5f}, Val L1 Loss: {val_loss.avg:.5f}")
        else:
            print(f"Epoch {e+1}/{epochs}: Train L1 Loss: {train_loss.avg:.5f}")
    
    return net_G

def train_gan(model, train_dl, val_dl=None, epochs=20, save_dir='models', 
             save_model=True, display_every=50, pretrained=False):
    """
    Train the GAN model
    
    Args:
        model (MainModel): Main GAN model
        train_dl (DataLoader): Training dataloader
        val_dl (DataLoader): Validation dataloader
        epochs (int): Number of training epochs
        save_dir (str): Directory to save models
        save_model (bool): Whether to save model checkpoints
        display_every (int): How often to display results
        pretrained (bool): Whether the generator is pretrained
        
    Returns:
        model (MainModel): Trained model
        loss_history (dict): Dictionary of loss histories
    """
    print(f"Training GAN for {epochs} epochs...")
    
    # Create directory for saving models
    if save_model:
        os.makedirs(save_dir, exist_ok=True)
    
    # Track loss history
    loss_history = {
        'loss_D_fake': [],
        'loss_D_real': [],
        'loss_D': [],
        'loss_G_GAN': [],
        'loss_G_L1': [],
        'loss_G': []
    }
    
    # Keep validation data for visualization
    val_data = next(iter(val_dl)) if val_dl else None
    
    # Train for specified number of epochs
    for e in range(epochs):
        # Create loss meters for tracking
        loss_meter_dict = create_loss_meters()
        
        # Training loop
        for i, data in enumerate(tqdm(train_dl, desc=f"Epoch {e+1}/{epochs}")):
            model.setup_input(data)
            model.optimize()
            
            # Update loss meters
            update_losses(model, loss_meter_dict, count=data['L'].size(0))
            
            # Display progress and visualize
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}, Iteration {i}/{len(train_dl)}")
                log_results(loss_meter_dict)
                
                if val_data is not None:
                    visualize(model, val_data, save=save_model, 
                             filename=os.path.join(save_dir, f"epoch_{e+1}_iter_{i}.png"))
        
        # Save model checkpoint
        if save_model and (e+1) % 5 == 0:  # Save every 5 epochs
            datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
            torch.save(model.net_G.state_dict(), 
                      os.path.join(save_dir, f"generator_{datetime_str}_epoch_{e+1}.pth"))
            torch.save(model.net_D.state_dict(), 
                      os.path.join(save_dir, f"discriminator_{datetime_str}_epoch_{e+1}.pth"))
        
        # Update loss history
        for loss_name, loss_meter in loss_meter_dict.items():
            loss_history[loss_name].append(loss_meter.avg)
        
        # Print epoch summary
        print(f"\nEpoch {e+1}/{epochs} completed")
        log_results(loss_meter_dict)
    
    # Save final model
    if save_model:
        datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        torch.save(model.net_G.state_dict(), 
                  os.path.join(save_dir, f"generator_final_{datetime_str}.pth"))
        torch.save(model.net_D.state_dict(), 
                  os.path.join(save_dir, f"discriminator_final_{datetime_str}.pth"))
    
    return model, loss_history

def setup_and_train(stack, pretrain_epochs=10, gan_epochs=20, batch_size=16, 
                   lambda_L1=100, save_dir='models', generator_type='resnet_unet'):
    """
    Setup and train the complete GAN model
    
    Args:
        stack (Stack): Stack object with data
        pretrain_epochs (int): Number of pretraining epochs
        gan_epochs (int): Number of GAN training epochs
        batch_size (int): Batch size for training
        lambda_L1 (float): Weight for L1 loss in GAN training
        save_dir (str): Directory to save models
        generator_type (str): Type of generator ('unet' or 'resnet_unet')
        
    Returns:
        stack (Stack): Updated stack with trained models
    """
    # Create PyTorch dataloaders
    train_dl, val_dl, test_dl = create_dataloaders(stack, batch_size)
    
    # Create generator and discriminator models
    net_G, net_D = stack.create_gan_models(
        input_channels=1, 
        output_channels=2, 
        model_type=generator_type
    )
    
    # Pretrain generator if specified
    if pretrain_epochs > 0:
        print(f"Pretraining generator ({generator_type}) for {pretrain_epochs} epochs...")
        net_G = pretrain_generator(net_G, train_dl, val_dl, epochs=pretrain_epochs)
    
    # Create full GAN model
    model = MainModel(
        net_G=net_G,
        net_D=net_D,
        lr_G=2e-4,
        lr_D=2e-4,
        lambda_L1=lambda_L1
    )
    
    # Train GAN
    model, loss_history = train_gan(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        epochs=gan_epochs,
        save_dir=save_dir,
        pretrained=(pretrain_epochs > 0)
    )
    
    # Update stack with trained models
    stack.net_G = model.net_G
    stack.net_D = model.net_D
    stack.finished_gan_training(loss_history)
    stack.save_gan_models(save_dir, "final")
    
    return stack

def train(stack, config=None):
    """
    Main training function to be called from outside this module
    
    Args:
        stack (Stack): Stack with data
        config (dict, optional): Configuration for training
        
    Returns:
        stack (Stack): Updated stack with trained models
    """
    if config is None:
        config = {
            'pretrain_epochs': 5,
            'gan_epochs': 10,
            'batch_size': 16,
            'lambda_L1': 100,
            'save_dir': 'models',
            'generator_type': 'resnet_unet'  # or 'unet'
        }
    
    stack = setup_and_train(
        stack=stack,
        pretrain_epochs=config['pretrain_epochs'],
        gan_epochs=config['gan_epochs'],
        batch_size=config['batch_size'],
        lambda_L1=config['lambda_L1'],
        save_dir=config['save_dir'],
        generator_type=config['generator_type']
    )
    
    return stack

# Entry point when run as a script
if __name__ == "__main__":
    # Create a new stack
    stack = Stack()
    
    # Load your data here
    # We need to initialize the data generators
    from objects.data_loader import load_data  # Import your data loading module
    
    # Initialize the generators with some sample data
    # If load_data function doesn't exist, you'll need to create it
    stack = load_data(stack)
    
    # Debugging: Check the state of the stack
    print(f"Stack object: {stack}")
    print(f"Train generator: {stack.train_generator}")
    
    # Fail early if generators aren't initialized
    if stack.train_generator is None or stack.val_generator is None or stack.test_generator is None:
        raise ValueError("Data generators must be initialized before training. Check your data loading code.")
    
    # Train GAN
    stack = train(stack) 