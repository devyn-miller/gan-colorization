import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

class UNetBlock(nn.Module):
    def __init__(self, nf, ni, submodule=None, input_c=None, dropout=False,
                 innermost=False, outermost=False):
        super().__init__()
        self.outermost = outermost
        if input_c is None: input_c = nf
        
        # Define the downsampling part of the block
        downconv = nn.Conv2d(input_c, ni, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(ni)
        
        # Define the upsampling part of the block
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(nf)
        
        if outermost:
            # Outermost block has special handling
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            # Innermost block doesn't have skip connections
            upconv = nn.ConvTranspose2d(ni, nf, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            # Middle blocks
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout: up += [nn.Dropout(0.5)]
            model = down + [submodule] + up
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            # Skip connections
            return torch.cat([x, self.model(x)], 1)

class UNet(nn.Module):
    def __init__(self, input_c=1, output_c=2, n_down=8, num_filters=64):
        """
        Standard U-Net Generator architecture.
        
        Args:
            input_c (int): Number of input channels (L channel of Lab color space)
            output_c (int): Number of output channels (a and b channels of Lab)
            n_down (int): Number of downsampling layers
            num_filters (int): Number of filters in the first conv layer
        """
        super().__init__()
        
        # Build the innermost block first, then build outward
        unet_block = UNetBlock(num_filters * 8, num_filters * 8, innermost=True)
        
        # Add the middle blocks with dropouts
        for _ in range(n_down - 5):
            unet_block = UNetBlock(num_filters * 8, num_filters * 8, submodule=unet_block, dropout=True)
        
        # Add the outer blocks with gradually decreasing number of filters
        out_filters = num_filters * 8
        for _ in range(3):
            unet_block = UNetBlock(out_filters // 2, out_filters, submodule=unet_block)
            out_filters //= 2
        
        # Add the outermost block
        self.model = UNetBlock(output_c, out_filters, input_c=input_c, submodule=unet_block, outermost=True)
    
    def forward(self, x):
        return self.model(x)

class ResNetUNet(nn.Module):
    def __init__(self, input_c=1, output_c=2, pretrained=True):
        """
        U-Net with ResNet18 encoder backbone.
        
        Args:
            input_c (int): Number of input channels (L channel of Lab color space)
            output_c (int): Number of output channels (a and b channels of Lab)
            pretrained (bool): Whether to use pretrained ResNet weights
        """
        super().__init__()
        
        # Load a pretrained ResNet model as the encoder
        if pretrained:
            # We'll modify the first conv layer to accept 1 channel
            resnet = models.resnet18(pretrained=True)
            
            # Modify the first layer to accept single channel
            if input_c != 3:
                old_conv = resnet.conv1
                new_conv = nn.Conv2d(input_c, 64, kernel_size=7, stride=2, padding=3, bias=False)
                # Initialize with average of RGB channels
                if pretrained:
                    with torch.no_grad():
                        new_conv.weight[:, :, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
                resnet.conv1 = new_conv
        else:
            resnet = models.resnet18(pretrained=False)
            if input_c != 3:
                resnet.conv1 = nn.Conv2d(input_c, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Use ResNet's layers as encoder parts (removing the final FC layer)
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64 channels
        self.encoder2 = nn.Sequential(resnet.maxpool, resnet.layer1)  # 64 channels
        self.encoder3 = resnet.layer2  # 128 channels
        self.encoder4 = resnet.layer3  # 256 channels
        self.encoder5 = resnet.layer4  # 512 channels
        
        # Decoder (upsampling) layers
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Final output layer
        self.final = nn.Sequential(
            nn.Conv2d(32, output_c, kernel_size=1),
            nn.Tanh()  # Use Tanh for -1 to 1 range
        )
    
    def forward(self, x):
        # Encoder path
        e1 = self.encoder1(x)  # [B, 64, 128, 128]
        e2 = self.encoder2(e1)  # [B, 64, 64, 64]
        e3 = self.encoder3(e2)  # [B, 128, 32, 32]
        e4 = self.encoder4(e3)  # [B, 256, 16, 16]
        e5 = self.encoder5(e4)  # [B, 512, 8, 8]
        
        # Decoder path with skip connections
        d1 = self.decoder1(e5)  # [B, 256, 16, 16]
        d1 = torch.cat([d1, e4], dim=1)  # [B, 512, 16, 16]
        
        d2 = self.decoder2(d1)  # [B, 128, 32, 32]
        d2 = torch.cat([d2, e3], dim=1)  # [B, 256, 32, 32]
        
        d3 = self.decoder3(d2)  # [B, 64, 64, 64]
        d3 = torch.cat([d3, e2], dim=1)  # [B, 128, 64, 64]
        
        d4 = self.decoder4(d3)  # [B, 64, 128, 128]
        d4 = torch.cat([d4, e1], dim=1)  # [B, 128, 128, 128]
        
        d5 = self.decoder5(d4)  # [B, 32, 256, 256]
        
        return self.final(d5)  # [B, output_c, 256, 256] 