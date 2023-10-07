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

  #flow structure
  flowDict={'WD120':[117.35140,9.32410],
            'WD150':[148.07293,9.32483],
            }
  for flow in ['WD120','WD150']:
    wd_input,ws_input=flowDict[flow]
    x_sample,y_sample=wind2xy(wd_input,ws_input)
    print(flow,x_sample,y_sample)
    vardata=model_ae.decode(torch.Tensor([x_sample,y_sample]).to(device)).detach().cpu().numpy()
    vardata=vardata[0,:,:,:]*thd
    np.save('data/VAE/flow_WD%.2f_WS%.2f.npy'%(wd_input,ws_input),vardata)
