#!/home/mileshsieh/anaconda3/bin/python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import dataUtils as du
import matplotlib.colors as mc
import matplotlib
import matplotlib.ticker as ticker
matplotlib.rc('xtick',labelsize=12)
matplotlib.rc('ytick',labelsize=12)

def plot_mu(plt,ax,lbl,var,mu,limit_value,vmin,vmax,cmap):    
    for tt in range(nt):
        plt.scatter(mu[:,tt,0],mu[:,tt,1],c=var,vmin=vmin,vmax=vmax,cmap=cmap)
    cb=plt.colorbar(extend='both')
    cb.set_label(pltCfg[lbl][1],fontsize=14)
    plt.grid(True)
    plt.title(pltCfg[lbl][0],fontsize=20)
    plt.xlim(-limit_value,limit_value)
    plt.ylim(-limit_value,limit_value)
    plt.xticks(np.arange(-limit_value,limit_value+1))
    plt.yticks(np.arange(-limit_value,limit_value+1))
    plt.xlabel('X',fontsize=14)
    plt.ylabel('Y',fontsize=14)
    return plt,ax

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

  #load the latent variables
  latent_mu_all=np.load('data/AE/latent/latent_var.%s.npy'%sf)
  print(latent_mu_all.shape)
  nCase,nt,_=latent_mu_all.shape
  #load the synoptic factors
  features=['wd925', 'ws925',]
  caseList=du.getCaseList(dataset)
  df=du.getSynoptic(caseList,features,dataset)
  ws=df.ws925.values
  wd=df.wd925.values
  print(ws.shape,wd.shape)

  # training by tt=6to60 (08:00 to 02:00) so that tt=0 means 08:00
  pltCfg={'WD':['Wind Direction','Wind Direction($^{\circ}$)'],
          'WS':['Wind Speed','Wind Speed(m/s)'],
        }

  limit_value=2
  fig=plt.figure(figsize=(14,6))
  ax1=plt.subplot(121)
  plt,ax1=plot_mu(plt,ax1,'WD',wd,latent_mu_all,limit_value,60,180,'Accent')
  ax2=plt.subplot(122)
  plt,ax1=plot_mu(plt,ax1,'WS',ws,latent_mu_all,limit_value,2,10,'Dark2')
  fig.subplots_adjust(top=0.85)
  plt.suptitle('Latent Space Color-coded by Synoptic Flow Regimes',fontsize=25)
  plt.annotate('(a)', xy=(0.05, 0.91), xytext=(0.05, 0.87),xycoords='figure fraction',fontsize=20)
  plt.annotate('(b)', xy=(0.05, 0.91), xytext=(0.49, 0.87),xycoords='figure fraction',fontsize=20)
  plt.savefig('figures/fig5_physical_latent_space.png',dpi=400)

