import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import lab2rgb
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

class GANLoss(nn.Module):
    """
    GAN loss module that can be used with different GAN modes.
    """
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        """
        Initialize the GANLoss class.
        
        Args:
            gan_mode (str): Type of GAN loss ('vanilla' or 'lsgan')
            real_label (float): Target value for real samples
            fake_label (float): Target value for fake samples
        """
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        
        if gan_mode == 'vanilla':
            # Standard binary cross entropy loss
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            # Least squares GAN loss
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported GAN mode: {gan_mode}")
    
    def get_labels(self, preds, target_is_real):
        """
        Create labels for the discriminator based on whether the prediction 
        should be real or fake.
        
        Args:
            preds (Tensor): Discriminator predictions
            target_is_real (bool): Whether the target is real (True) or fake (False)
        
        Returns:
            Tensor: Labels of the same shape as preds
        """
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)
    
    def __call__(self, preds, target_is_real):
        """
        Calculate the GAN loss.
        
        Args:
            preds (Tensor): Discriminator predictions
            target_is_real (bool): Whether the target is real (True) or fake (False)
        
        Returns:
            Tensor: Computed loss
        """
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss

class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count

def create_loss_meters():
    """
    Create a dictionary of AverageMeter for tracking losses.
    
    Returns:
        dict: Dictionary of loss meters
    """
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()
    
    return {
        'loss_D_fake': loss_D_fake,
        'loss_D_real': loss_D_real,
        'loss_D': loss_D,
        'loss_G_GAN': loss_G_GAN,
        'loss_G_L1': loss_G_L1,
        'loss_G': loss_G
    }

def update_losses(model, loss_meter_dict, count):
    """
    Update the loss meters with the current losses.
    
    Args:
        model: The GAN model
        loss_meter_dict (dict): Dictionary of loss meters
        count (int): Number of samples
    """
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)

def lab_to_rgb(L, ab):
    """
    Convert a batch of L and ab channels to RGB images.
    
    Args:
        L (Tensor): Lightness channel tensor [B, 1, H, W]
        ab (Tensor): a and b channel tensor [B, 2, H, W]
    
    Returns:
        np.array: RGB images [B, H, W, 3]
    """
    # Convert tensors to numpy arrays and scale values
    L = (L + 1.) * 50.  # Scale L channel from [-1, 1] to [0, 100]
    ab = ab * 110.      # Scale ab channels from [-1, 1] to [-110, 110]
    
    # Combine L and ab channels
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    
    # Convert each Lab image to RGB
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    
    return np.stack(rgb_imgs, axis=0)

def visualize(model, data, save=True, filename=None):
    """
    Visualize the model's input, output, and ground truth.
    
    Args:
        model: The GAN model
        data (dict): Batch of data with 'L' and 'ab' keys
        save (bool): Whether to save the visualization
        filename (str, optional): Filename to save the visualization
    """
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    
    # Get fake (generated) colors, real colors, and L channel
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    
    # Convert to RGB for visualization
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    
    # Create a figure for visualization
    fig = plt.figure(figsize=(15, 8))
    
    # Number of images to show
    num_images = min(5, L.size(0))
    
    # Plot L channel, fake color, and real color for each image
    for i in range(num_images):
        ax = plt.subplot(3, num_images, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax.set_title("Input (L)")
        
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax.set_title("Generated")
        
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        ax.imshow(real_imgs[i])
        ax.axis("off")
        ax.set_title("Ground Truth")
    
    plt.tight_layout()
    plt.show()
    
    # Save the figure if requested
    if save and filename:
        fig.savefig(filename)

def log_results(loss_meter_dict):
    """
    Print the average of each loss meter.
    
    Args:
        loss_meter_dict (dict): Dictionary of loss meters
    """
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")

def calculate_metrics(pred_rgb, target_rgb):
    """
    Calculate PSNR and SSIM metrics between prediction and target.
    
    Args:
        pred_rgb (np.ndarray): Predicted RGB image
        target_rgb (np.ndarray): Target RGB image
    
    Returns:
        tuple: (psnr, ssim) metrics
    """
    # Calculate PSNR
    psnr = psnr_metric(target_rgb, pred_rgb)
    
    # Calculate SSIM (over each channel and take the mean)
    # Fix: Use win_size=5 (smaller window) and specify channel_axis for small images
    # The default win_size is 7, which might be too large for small images
    ssim = ssim_metric(
        target_rgb, 
        pred_rgb, 
        channel_axis=2,  # Specify the channel axis (RGB is the last dimension)
        data_range=1.0,
        win_size=5  # Use smaller window size for small images
    )
    
    return psnr, ssim

def init_weights(net, init='norm', gain=0.02):
    """
    Initialize network weights.
    
    Args:
        net (nn.Module): Network to initialize
        init (str): Initialization method
        gain (float): Gain parameter for initialization
    
    Returns:
        nn.Module: Initialized network
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)
    
    net.apply(init_func)
    print(f"Model initialized with {init} initialization")
    return net 