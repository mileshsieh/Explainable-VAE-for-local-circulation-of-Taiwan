#!/home/mileshsieh/anaconda3/bin/python

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from VAE_uv2uv_61x61 import variationalAutoEncoder,Encoder,Decoder,Reshape
import dataUtils as du
from scipy.optimize import fsolve
import matplotlib
import matplotlib.colors as mc
import matplotlib.ticker as ticker
matplotlib.rc('xtick',labelsize=25)
matplotlib.rc('ytick',labelsize=25)

[a,b,x0,y0]=[0.0531645,0.89609678,10.06394176,-0.02435647]
[a0,a1,wd0]=[2.506213,-0.99329623,108.75227799]

def ws_h(x, y):
    return a * (x - x0)**2 + b * (y - y0)**2

def wd_h(x, y):
    return a1*np.arctan(y/(x-a0))/np.pi*180+wd0

def wind2xy(wd,ws):
    def equations(vars):
        x, y = vars
        alpha=np.tan((wd-wd0)*np.pi/180/a1)
        eq1 = x-a0-y/alpha
        eq2 = a*(x-x0)**2+b*(y-y0)**2-ws
        return [eq1, eq2]

    x, y =  fsolve(equations, (2.1, 0))
    return x,y

def plotWSWDAxes(ax,clr_WD,clr_WS):
    mu_x=np.linspace(-4,2,100)
    mu_y=np.linspace(-3,3,100)
    grid_x,grid_y=np.meshgrid(mu_x,mu_y)
    c_wd=ax.contour(grid_x,grid_y,wd_h(grid_x,grid_y),colors=[clr_WD])
    ax.clabel(c_wd, levels=[40,80,140], inline=True, fmt='$WD=%.0f\degree$', fontsize=30)

    c_ws=ax.contour(grid_x,grid_y,ws_h(grid_x,grid_y),colors=[clr_WS])
    ax.clabel(c_ws, levels=[6,10,12], inline=True, fmt='WS=%.1f m/s', fontsize=30)
    #plt.xticks([])
    #plt.yticks([])
    return

def plotStreamLine(ax,topo,vardata,title,rtitle,ltitle):
    ny,nx=topo.shape
    xx,yy=np.meshgrid(np.arange(nx),np.arange(ny))
    ax.contour(topo,levels=[0.05,],colors=['k'])
    #ax.contour(topo,levels=[0.05,0.2],colors=['k'])
    ws=np.sqrt(vardata[0,:,:]**2+vardata[1,:,:]**2)
    #print(ws.min(),ws.max())
    strm=ax.streamplot(xx,yy,vardata[0,:,:],vardata[1,:,:],
            color=ws,cmap='YlGnBu',norm=mc.Normalize(vmin=0.0,vmax=7.0),density=0.6,linewidth=2.2,arrowsize=2,zorder=3)
    #ax.plot([xstart,xend,xend,xstart,xstart],[ystart,ystart,yend,yend,ystart],lw=2,ls='--')
    ax.contourf(topo,levels=[0.2,5.0],colors=['darkgreen'],zorder=5)
    ax.set_xlim(0,nx)
    ax.set_ylim(0,ny)
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.set_title(rtitle,fontsize=10,loc='right')
    ax.set_title(ltitle,fontsize=10,loc='left')
    ax.set_title(title,fontsize=25,loc='center')
    return strm

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

  device = torch.device("cuda:0")

  # Load model
  model_ae = torch.load('data/VAE/leevortex.%s.pth'%sf,map_location=torch.device('cuda:0'))
  model_ae.eval()

  #sample latent space according to the wd/ws
  wdList=[160,120,100,60]
  wsList=[10,8,6,4]
  nx_ldim=len(wsList)
  ny_ldim=len(wdList)
  nvar=2 # U and V
  topo=du.load_topo(du.ys,du.ye,du.xs,du.xe,step=5)
  ny,nx=topo.shape
  pattern=np.zeros((ny_ldim,nx_ldim,nvar,ny,nx))
  xList=[]
  yList=[]

  for ii in range(len(wdList)):
    for jj in range(len(wsList)):
      x_temp,y_temp=wind2xy(wdList[ii],wsList[jj])
      xList.append(x_temp)
      yList.append(y_temp)
      pattern[ii,jj,:,:,:]=model_ae.decode(torch.Tensor([x_temp,y_temp]).to(device)).detach().cpu().numpy()

  #it shound multiple thd to scale back to the original U and V valuse
  pattern=pattern*thd 
  print(pattern.shape)
  ny_ldim,nx_ldim,nvar,ny,nx=pattern.shape
  print(ny_ldim,nx_ldim,nvar,ny,nx)
  chList=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p']
  plt.close()
  fig, axes = plt.subplots(ncols=nx_ldim,nrows=ny_ldim,constrained_layout=False,figsize=((int(nx_ldim*4*2),ny_ldim*4)))
  kk=0
  for ii in range(nx_ldim):
    for jj in range(ny_ldim):
      strm=plotStreamLine(axes[ii,jj],topo,pattern[ii,jj,:,:,:],'%s [wd=%.0f,ws=%.0f]'%(chList[kk],wdList[ii],wsList[jj]),'','')
      kk=kk+1
  
  fig.subplots_adjust(left=0.45,right=0.88)
  ax_cb = fig.add_axes([0.9, 0.1, 0.02, 0.8])
  cbar=plt.colorbar(strm.lines,cax=ax_cb,extend='max')
  cbar.set_label('Wind Speed (m/s)',fontsize=35)

  #physical latent space
  ax=fig.add_axes([0.05,0.2,0.35,0.6])

  #for specific flow regime
  plotWSWDAxes(ax,'darkred','blue')

  for ch,(sample_x,sample_y) in zip(chList,zip(xList,yList)):
    ax.plot(sample_x,sample_y,marker='o',markersize=20,color='g')
    ax.text(sample_x,sample_y, ch+' ', va='top', ha='right',fontsize=35)  
     #plt=plot_sample(plt,sampling[rgm][0],sampling[rgm][1],'g',rgm)
    
  ax.set_title('Physical Latent Space',fontsize=40)
  plt.suptitle('Various Generated Local Circulations Corresonding to Synoptic Flow Regime Change',fontsize=50)

  plt.annotate('(a)', xy=(0.05, 0.87), xytext=(0.05, 0.87),xycoords='figure fraction',fontsize=40)
  plt.annotate('(b)', xy=(0.41, 0.87), xytext=(0.41, 0.87),xycoords='figure fraction',fontsize=40)

  plt.savefig('figures/fig6_physical_generation.png',dpi=400)





