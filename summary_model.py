#!/home/mileshsieh/anaconda3/bin/python
import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary

#input/output: 2x300x300
class Reshape(nn.Module):
  def __init__(self, *args):
    super().__init__()
    self.shape = args

  def forward(self, x):
    return x.view(self.shape)

class Encoder(nn.Module):

  def __init__(self,num_input_channels,latent_dim,act_fn,c_hid):
    """
    Inputs:
    -num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
    -base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
    -latent_dim : Dimensionality of latent representation z
    -act_fn : Activation function used throughout the encoder network
    """
    super().__init__()
    #input should be (2,61,61) downsampled from (2,301,301)
    self.net = nn.Sequential(
      #input should be (2,61,61) downsampled from (2,301,301)
      nn.Conv2d(num_input_channels, c_hid*4, kernel_size=3, stride=1), #58
      nn.MaxPool2d(kernel_size=2, stride=1), #57
      act_fn(),
      nn.Conv2d(c_hid*4, c_hid*8, kernel_size=3, stride=1), #55
      act_fn(),
      nn.Conv2d(c_hid*8, c_hid*32, kernel_size=3, stride=2), #27
      act_fn(),
      nn.Conv2d(c_hid*32, c_hid*64, kernel_size=3, stride=1), #25
      nn.MaxPool2d(kernel_size=2, stride=2), #12
      act_fn(),
      nn.Conv2d(c_hid*64, c_hid*128, kernel_size=3, stride=2), #5
      act_fn(),
      nn.Flatten(), # Image grid to single feature vector
      
    )
  def forward(self, x):
    return self.net(x)
    
class Decoder(nn.Module):

  def __init__(self,num_input_channels,latent_dim,act_fn,c_hid):
    """
    Inputs:
    -num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
    -base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
    -latent_dim : Dimensionality of latent representation z
    -act_fn : Activation function used throughout the decoder network
    """
    super().__init__()

    self.net = nn.Sequential(
      #encoder compress (2,61,61) into (2,c_hid*128,5,5)
      #nn.Linear(latent_dim,5*5*c_hid*2),
      Reshape(-1, c_hid*128, 5, 5),
      nn.ConvTranspose2d(c_hid*128, c_hid*64, kernel_size=3, stride=2), #11
      nn.Upsample(scale_factor=2), #22
      act_fn(),
      nn.ConvTranspose2d(c_hid*64, c_hid*32, kernel_size=3, stride=1), #24
      act_fn(),
      nn.ConvTranspose2d(c_hid*32, c_hid*8, kernel_size=3, stride=2), #49
      act_fn(),
      nn.ConvTranspose2d(c_hid*8, c_hid*4, kernel_size=5, stride=1), #53
      act_fn(),
      nn.ConvTranspose2d(c_hid*4, num_input_channels, kernel_size=5, stride=1), #57
      act_fn(),
      nn.ConvTranspose2d(num_input_channels, num_input_channels, kernel_size=5, stride=1), #61
      
    )

  def forward(self, x):
    return self.net(x)

class variationalAutoEncoder(nn.Module):
  def __init__(self, num_input_channels,latent_dim,act_fn,c_hid=2):
    super(variationalAutoEncoder, self).__init__()
    # Encoder
    self.encoder = Encoder(num_input_channels,latent_dim,act_fn,c_hid)
    # Decoder
    self.decoder = Decoder(num_input_channels,latent_dim,act_fn,c_hid)
    self.fc1 = nn.Linear(5*5*c_hid*128, latent_dim)
    self.fc2 = nn.Linear(5*5*c_hid*128, latent_dim)
    self.fc3 = nn.Linear(latent_dim, 5*5*c_hid*128)

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5*logvar) # standard deviation
    eps = torch.randn_like(std) # `randn_like` as we need the same size
    sample = mu + (eps * std) # sampling as if coming from the input spac

    #std = logvar.mul(0.5).exp_()
    #esp = torch.randn(*mu.size()).to(device)
    #z = mu + std * esp
    #return z
    return sample
    
  def bottleneck(self, h):
    mu, logvar = self.fc1(h), self.fc2(h)
    z = self.reparameterize(mu, logvar)
    return z, mu, logvar

  def encode(self, x):
    h = self.encoder(x)
    z, mu, logvar = self.bottleneck(h)
    return z, mu, logvar

  def decode(self, z):
    z = self.fc3(z)
    z = self.decoder(z)
    return z

  def forward(self, x):
    z, mu, logvar = self.encode(x)
    z = self.decode(z)
    return z, mu, logvar
    
def loss_fn(recon_x, x, mu, logvar, beta):
    MSE = torch.nn.functional.mse_loss(recon_x,x)
    KLD = -0.05 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD*beta , MSE, KLD

if __name__=='__main__':
  num_input_channels=2
  latent_dim=2

  device = torch.device("cuda:0")


  model_vae = variationalAutoEncoder(num_input_channels,latent_dim,nn.Tanh).to(device)

  summary(model_vae,(2,61,61))
