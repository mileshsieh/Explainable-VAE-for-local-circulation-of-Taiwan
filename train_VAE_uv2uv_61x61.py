#!/home/mileshsieh/anaconda3/bin/python
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import numpy as np
import glob,random,os
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import dataUtils as du

#for reproducibility
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

seed=3
def set_seed(seed, loader=None):
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  #try:
  #    loader.sampler.generator.manual_seed(seed)
  #except AttributeError:
  #    pass

set_seed(seed)

def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % 2**32
  numpy.random.seed(worker_seed)
  random.seed(worker_seed)

torch.use_deterministic_algorithms(True)
g = torch.Generator()
g.manual_seed(seed)

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
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD*beta , MSE, KLD

if __name__=='__main__':
  num_input_channels=2
  latent_dim=3
  beta=0.01
  #for normalized
  n_p=99

  #for training
  epochs = 1000
  epo_print=10
  batch_size = 40
  lr = 0.001

  dataset='ctrl'

  #for write out
  sf='vae61x61_ldim%d_b%.4f_%s_t%dto%d_seed%d'%(latent_dim,beta,dataset,du.ts,du.te,seed)


  device = torch.device("cuda:1")

  #load data
  caseList,X=du.load_dataset(dataset)
  print(X.shape)
  
  #scale
  nt,ncase,nvar,ny,nx=X.shape
  thd=np.percentile(abs(X),n_p)
  sf='%s_norm%d'%(sf,int(thd))
  print('threshold=%.2f model name:%s'%(thd,sf))
  X=(X/thd).reshape(nt*ncase,nvar,ny,nx)

  #split dataset
  indices=torch.randperm(nt*ncase,generator=g)

  test_size=int(nt*ncase*0.2)
  X_test=X[indices[:test_size]]
  X_train=X[indices[test_size:]]
  #save indices for validation, CAE trainging, and PCA
  np.save('./data/VAE/input/testing_indices.npy',indices[:test_size])
  np.save('./data/VAE/input/training_indices.npy',indices[test_size:])

  print(X_train.shape,X_test.shape)
  # dataloader
  train_loader=DataLoader(dataset=torch.from_numpy(X_train).float(),batch_size=batch_size,shuffle=True)
  test_loader=DataLoader(dataset=torch.from_numpy(X_test).float(),batch_size=batch_size,shuffle=True)


  model_vae = variationalAutoEncoder(num_input_channels,latent_dim,nn.Tanh).to(device)
  optimizer = torch.optim.Adam(model_vae.parameters(), lr=lr)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,40], gamma=0.5)

  # Train
  vae_loss=[]
  vae_kld=[]
  mse_train=[]
  mse_test=[]

  for epoch in range(epochs):
    total_loss=0
    total_mse=0
    total_kld=0
    model_vae.train()
    for data in train_loader:
      inputs = data.to(device) 
      model_vae.zero_grad()
      # Forward
      recon, mu, logvar = model_vae(inputs)
      loss, mse, kld = loss_fn(recon, inputs, mu, logvar,beta)
      loss.backward()
      optimizer.step()
      total_loss+=float(loss)
      total_mse+=float(mse)
      total_kld+=float(kld)

    total_loss /= len(train_loader.dataset)
    total_mse /= len(train_loader.dataset)
    total_kld /= len(train_loader.dataset)

    # testing mse
    total_test_mse=0
    model_vae.eval()
    with torch.no_grad():
      for data in test_loader:
        inputs_test = data.to(device) 
        recon_test, _, _ = model_vae(inputs_test)
        test_mse = torch.nn.functional.mse_loss(recon_test,inputs_test)
        total_test_mse+=float(test_mse)
      total_test_mse /= len(test_loader.dataset)

    scheduler.step()
    if epoch % epo_print ==0:
      print('[{}/{}] Loss:'.format(epoch+1, epochs), total_loss,total_mse,total_test_mse,total_kld)
      vae_loss.append(total_loss)
      vae_kld.append(total_kld)
      mse_train.append(total_mse)
      mse_test.append(total_test_mse)
  np.save('data/VAE/loss/vae_loss.%s.npy'%sf,np.array(vae_loss))
  np.save('data/VAE/loss/vae_kld.%s.npy'%sf,np.array(vae_kld))
  np.save('data/VAE/loss/mse_train.%s.npy'%sf,np.array(mse_train))
  np.save('data/VAE/loss/mse_test.%s.npy'%sf,np.array(mse_test))
  torch.save(model_vae, 'data/VAE/model/leevortex.%s.pth'%sf)
