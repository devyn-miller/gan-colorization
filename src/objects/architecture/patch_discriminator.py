import torch
import torch.nn as nn

class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator for conditional image-to-image translation.
    This discriminator looks at patches of the image and classifies them as real/fake.
    The conditional aspect means it takes both the input (L channel) and output (ab channels)
    or the real target and discriminates the combined image.
    """
    def __init__(self, input_c=3, num_filters=64, n_down=3):
        """
        Args:
            input_c (int): Number of input channels (3 for L+ab channels combined)
            num_filters (int): Number of filters in the first conv layer
            n_down (int): Number of downsampling operations
        """
        super().__init__()
        
        # Build the discriminator model as a sequence of Conv-BatchNorm-LeakyReLU blocks
        model = [self.get_layers(input_c, num_filters, norm=False)]
        
        # Add downsampling blocks
        # Note: the last block uses stride=1
        model += [
            self.get_layers(
                num_filters * 2 ** i, 
                num_filters * 2 ** (i + 1), 
                s=1 if i == (n_down-1) else 2
            ) 
            for i in range(n_down)
        ]
        
        # Add the final layer with no normalization or activation
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False, act=False)]
        
        self.model = nn.Sequential(*model)
    
    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True):
        """
        Creates a sequence of Conv2d + optional BatchNorm + optional LeakyReLU
        
        Args:
            ni (int): Number of input channels
            nf (int): Number of output channels
            k (int): Kernel size
            s (int): Stride
            p (int): Padding
            norm (bool): Whether to include BatchNorm
            act (bool): Whether to include LeakyReLU activation
        """
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (Tensor): Input tensor of shape [B, input_c, H, W]
                        This is the concatenated L and ab channels
        
        Returns:
            Tensor: Patch predictions of shape [B, 1, H/2^n_down, W/2^n_down]
        """
        return self.model(x) 