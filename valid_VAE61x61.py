import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from VAE_uv2uv_61x61 import variationalAutoEncoder,Encoder,Decoder,Reshape
from sklearn.preprocessing import StandardScaler
import dataUtils as du
from itertools import permutations
import matplotlib.colors as mc
import matplotlib
import matplotlib.ticker as ticker
matplotlib.rc('xtick',labelsize=25)
matplotlib.rc('ytick',labelsize=25)

def plotVar(ax,vardata,title,rtitle,ltitle):
    #ax.contourf(vardata,vmin=-3,vmax=3,cmap='bwr')
    ax.imshow(vardata[::-1,:],vmin=-3,vmax=3,cmap='bwr')
    ax.contour(topo[::-1,:],levels=[0.5,],colors=['k',])
    ax.set_title(rtitle,fontsize=14,loc='right')
    ax.set_title(ltitle,fontsize=14,loc='left')
    ax.set_title(title,fontsize=14,loc='center')
    return

def plotStreamLine(ax,topo,vardata,title,rtitle,ltitle):
    ny,nx=topo.shape
    xx,yy=np.meshgrid(np.arange(nx),np.arange(ny))
    ax.contour(topo,levels=[0.05,],colors=['k'])
    #ax.contour(topo,levels=[0.05,0.2],colors=['k'])
    ws=np.sqrt(vardata[0,:,:]**2+vardata[1,:,:]**2)
    print(ws.min(),ws.max())
    strm=ax.streamplot(xx,yy,vardata[0,:,:],vardata[1,:,:],
            color=ws,cmap='viridis',norm=mc.Normalize(vmin=0.0,vmax=15.0),density=1.4,zorder=3)
    #ax.plot([xstart,xend,xend,xstart,xstart],[ystart,ystart,yend,yend,ystart],lw=2,ls='--')
    ax.contourf(topo,levels=[0.2,5.0],colors=['darkgreen'],zorder=5)
    ax.set_xlim(0,nx)
    ax.set_ylim(0,ny)
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.set_title(rtitle,fontsize=20,loc='right')
    ax.set_title(ltitle,fontsize=20,loc='left')
    ax.set_title(title,fontsize=20,loc='center')
    return strm

if __name__=='__main__':
  num_input_channels=2
  latent_dim=2
  beta=0.1
  use_cuda = 1
  #for activation function
  actn='tanh'
  batch_size = 48
  dataset='ctrl'
  thd=11

  #thd=13

  c_hid=1

  seed=3
  #for write out
  sf='vae61x61_ldim%d_b%.4f_%s_t%dto%d_seed%d_norm%d'%(latent_dim,beta,dataset,du.ts,du.te,seed,thd)

  device = torch.device('cuda:0')

  # Load model
  model_ae = torch.load('leevortex.%s.pth'%sf,map_location=torch.device('cuda:0'))
  model_ae.eval()

  # test by each run
  #load data
  X,topo,caseList=du.load_leevortex_data(du.ts,du.te,du.ys,du.ye,du.xs,du.xe,dataset,scaled=False,reshape=False,step=5)
  print(X.shape)

  #scale
  nt,ncase,nvar,ny,nx=X.shape
  input_X=(X/thd).reshape(nt,ncase,nvar,ny,nx)

  latent_mu_all=[]
  latent_logvar_all=[]
  for icase,case in enumerate(caseList):
    test_data=torch.from_numpy(input_X[:,icase,:,:,:]).float()  
    ntest=test_data.shape[0]
    #  dataloader
    test_loader = DataLoader(dataset = test_data, batch_size = ntest, shuffle = False)

    # Test
    with torch.no_grad():
      inputs = test_data
      #output of vae is (recon, mu, logvar)
      outputs,mu,logvar = model_ae(inputs.to(device))
    output_X = outputs.detach().cpu().numpy()
    output_X=output_X*thd
    np.save('data/VAE/reconstruction/pred.%s.VAE.%s.npy'%(case,sf),output_X)

    latent_mu=mu.detach().cpu().numpy()
    latent_mu_all.append(latent_mu)
    latent_logvar=logvar.detach().cpu().numpy()
    latent_logvar_all.append(latent_logvar)
    print(case,latent_mu.shape,latent_logvar.shape)
    
   
  latent_mu_all=np.array(latent_mu_all)
  print(latent_mu_all.shape)
  np.save('data/VAE/latent_mu_all.%s.npy'%sf,latent_mu_all)
  latent_logvar_all=np.array(latent_logvar_all)
  print(latent_logvar_all.shape)
  np.save('data/VAE/latent_logvar_all.%s.npy'%sf,latent_logvar_all)
  
  #get synoptic factors
  features=['wd925', 'ws925']
  df=du.getSynoptic(caseList,features,dataset)
  ws=df.ws925.values
  wd=df.wd925.values
  print(ws.shape,wd.shape)

  # training by tt=6to60 (08:00 to 02:00) so that tt=0 means 08:00
  pltCfg={'WD':['Wind Direction (deg)','WD(deg)'],
          'WS':['Wind Speed (m/s)','WS(m/s)'],
         }
  for lbl,var in zip(['WD','WS'],[wd,ws]):
    
    #plot each 2D relationship in latent space
    for (v1,v2) in permutations(np.arange(latent_mu_all.shape[2]),2):
      plt.close()
      plt.figure(figsize=(16,16))
      for tt in range(nt):
          if tt in du.skip_tt:
              plt.scatter(latent_mu_all[:,tt,v1],latent_mu_all[:,tt,v2],c=var,marker='x')
          else:
              plt.scatter(latent_mu_all[:,tt,v1],latent_mu_all[:,tt,v2],c=var,marker='o')
      cb=plt.colorbar()
      cb.set_label(pltCfg[lbl][1],fontsize=35)
      plt.grid(True)
      plt.title(pltCfg[lbl][0],fontsize=35)
      plt.xlabel('var[%d]'%v1,fontsize=35)
      plt.ylabel('var[%d]'%v2,fontsize=35)
      plt.savefig('figures/latent_features.%s.%s.%dx%d.png'%(lbl,sf,v1,v2))
