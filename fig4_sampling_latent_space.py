import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from VAE_uv2uv_61x61 import variationalAutoEncoder,Encoder,Decoder,Reshape
import dataUtils as du
import matplotlib.colors as mc
import matplotlib
import matplotlib.ticker as ticker
matplotlib.rc('xtick',labelsize=20)
matplotlib.rc('ytick',labelsize=20)

def plotStreamLine(ax,topo,vardata,title,rtitle,ltitle):
    ny,nx=topo.shape
    xx,yy=np.meshgrid(np.arange(nx),np.arange(ny))
    ax.contour(topo,levels=[0.05,],colors=['k'])
    #ax.contour(topo,levels=[0.05,0.2],colors=['k'])
    ws=np.sqrt(vardata[0,:,:]**2+vardata[1,:,:]**2)
    #print(ws.min(),ws.max())
    strm=ax.streamplot(xx,yy,vardata[0,:,:],vardata[1,:,:],
            color=ws,cmap='YlGnBu',norm=mc.Normalize(vmin=0.0,vmax=7.0),density=0.6,linewidth=2.3,arrowsize=2,zorder=3)
    #ax.plot([xstart,xend,xend,xstart,xstart],[ystart,ystart,yend,yend,ystart],lw=2,ls='--')
    ax.contourf(topo,levels=[0.2,5.0],colors=['darkgreen'],zorder=5)
    ax.set_xlim(0,nx)
    ax.set_ylim(0,ny)
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.set_title(rtitle,fontsize=10,loc='right')
    ax.set_title(ltitle,fontsize=10,loc='left')
    ax.set_title(title,fontsize=20,loc='center')
    return strm

def plotPatternPhase(xx,yy,pattern,title):
    ny_ldim,nx_ldim,nvar,ny,nx=pattern.shape
    print(ny_ldim,nx_ldim,nvar,ny,nx)
    title_ch=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p']
    plt.close()
    fig, axes = plt.subplots(ncols=nx_ldim,nrows=ny_ldim,constrained_layout=False,figsize=((nx_ldim*4+1,ny_ldim*4)))
    for ii in range(nx_ldim):
        for jj in range(ny_ldim):
            strm=plotStreamLine(axes[jj,ii],topo,pattern[ny_ldim-jj-1,ii,:,:,:],'%s (%.1f,%.1f)'%(title_ch[jj*nx_ldim+ii],xx[ny_ldim-jj-1,ii],yy[ny_ldim-jj-1,ii]),'','')

    fig.subplots_adjust(right=0.85)
    ax_cb = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    cbar=plt.colorbar(strm.lines,cax=ax_cb,extend='max')
    cbar.set_label('Wind Speed (m/s)',fontsize=25)
    plt.suptitle(title,fontsize=40)
    return plt.gca()

def plot_mu_mono(mu):
    ncase,nt,ndim=mu.shape
    plt.close()
    plt.figure(figsize=(9,7.5))
    plt.scatter(mu[:,:,0],mu[:,:,1],c='grey')
    plt.grid(True)
    plt.title('latent space',fontsize=25)
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.xticks(np.arange(-2,3))
    plt.yticks(np.arange(-2,3))
    #plt.xticks([])
    #plt.yticks([])
    #plt.xlabel('var[0]',fontsize=25)
    #plt.ylabel('var[1]',fontsize=25)
    return plt.gca()

if __name__=='__main__':
  num_input_channels=2
  latent_dim=2
  beta=0.1
  use_cuda = 1

  batch_size = 48
  dataset='ctrl'
  thd=11
  seed=3
  #for write out
  sf='vae61x61_ldim%d_b%.4f_%s_t%dto%d_seed%d_norm%d'%(latent_dim,beta,dataset,du.ts,du.te,seed,thd)

  device = torch.device('cuda:0')

  # Load model
  model_ae = torch.load('data/VAE/leevortex.%s.pth'%sf,map_location=torch.device('cuda:0'))
  model_ae.eval()

  #load the latent variables
  latent_mu_all=np.load('data/VAE/latent_mu_all.%s.npy'%sf)
  print(latent_mu_all.shape)
  nCase,nt,_=latent_mu_all.shape
  #load the synoptic factors
  features=['wd925', 'ws925']
  caseList=du.getCaseList(dataset)
  df=du.getSynoptic(caseList,features,dataset)
  ws=df.ws925.values
  wd=df.wd925.values

  # training by tt=6to60 (08:00 to 02:00) so that tt=0 means 08:00
  pltCfg={'WD':['Wind Direction (deg)','WD(deg)'],
          'WS':['Wind Speed (m/s)','WS(m/s)'],
        }
  
  xx,yy=np.meshgrid(np.arange(-1.5,1.6,1),np.arange(-1.5,1.6,1))
  ny_ldim,nx_ldim=xx.shape

  #plot
  nvar=2 # U and V
  topo=du.load_topo(du.ys,du.ye,du.xs,du.xe,step=5)
  ny,nx=topo.shape
  pattern=np.zeros((ny_ldim,nx_ldim,nvar,ny,nx))
  
  for ii in range(nx_ldim):
      for jj in range(ny_ldim):
      #for jj in [0,-1]:
          pattern[jj,ii,:,:,:]=model_ae.decode(torch.Tensor([xx[jj,ii],yy[jj,ii]]).to(device)).detach().cpu().numpy()
  
  #it shound multiple thd to scale back to the original U and V valuse
  pattern=pattern*thd 
  print(pattern.shape)
  ax=plotPatternPhase(xx,yy,pattern,'Generated Local Circulation')
  plt.savefig('figures/fig4_generated_local_circulation.png',dpi=400)





