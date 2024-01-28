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
matplotlib.rc('xtick',labelsize=12)
matplotlib.rc('ytick',labelsize=12)

[a,b,x0,y0]=[0.0531645,0.89609678,10.06394176,-0.02435647]
[a0,a1,wd0]=[2.506213,-0.99329623,110-1.24772201]

def ws_h(x, y):
    return a * (x - x0)**2 + b * (y - y0)**2

def wd_h(x, y):
    #return theta[0]*(x-theta[2])**2 - theta[1]*y
    return a1*np.arctan(y/(x-a0))/np.pi*180+wd0
def xy2ws(x,y):
    return 0.0531645*(x-10.06394176)**2+0.89609678*(y+0.02435647)**2
#wind direction
def xy2wd(x,y):
    return -0.99329623*np.arctan(y/(x-2.506213))/np.pi*180-1.24772201+110

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
    ax.clabel(c_wd, levels=[40,80,140], inline=True, fmt='$WD=%.0f\degree$', fontsize=12)

    c_ws=ax.contour(grid_x,grid_y,ws_h(grid_x,grid_y),colors=[clr_WS])
    ax.clabel(c_ws, levels=[6,10,12], inline=True, fmt='WS=%.1f m/s', fontsize=12)
    #plt.xticks([])
    #plt.yticks([])
    return

cities={'Taipei':[121.5598,25.09108],'Kaohsiung':[120.311922,22.620856]}

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
  model_ae = torch.load('./data/VAE/leevortex.%s.pth'%sf,map_location=torch.device('cuda:0'))
  model_ae.eval()

  #load the flow regimes in CMIP model
  m='TaiESM1'
  xbins=np.linspace(-4,2,7)
  ybins=np.linspace(-3,3,7)
  cntData=np.load('./data/CMIP6/synoptic_regime_days.%s.npy'%m)

  #sample latent space according to the wd/ws
  xList=[1.5,1.5]
  yList=[2.5,-0.5]
  n_sample=2 #2 samples
  nvar=2 # U and V
  topo=du.load_topo(du.ys,du.ye,du.xs,du.xe,step=5)
  ys=150
  ye=451
  xs=20
  xe=321
  lonTW=np.load('./data/tw_s_lon.npy')[xs:xe]
  latTW=np.load('./data/tw_s_lat.npy')[ys:ye]
  topo=np.load('./data/topodata.npy')[ys:ye,xs:xe]
  topo_m=np.copy(topo)
  topo_m[topo_m==0]=np.nan

  xx,yy=np.meshgrid(lonTW[::5],latTW[::5])

  ny,nx=xx.shape
  flow=np.zeros((n_sample,nvar,ny,nx))

  wdList=[]
  wsList=[]
  for i in range(len(xList)):
    ws_sample=xy2ws(xList[i],yList[i])  
    wd_sample=xy2wd(xList[i],yList[i])  
    wsList.append(ws_sample)
    wdList.append(wd_sample)
    flow[i,:,:,:]=model_ae.decode(torch.Tensor([xList[i],yList[i]]).to(device)).detach().cpu().numpy()
    print('sample %d:ws=%.2f wd=%.2f'%(i+1,ws_sample,wd_sample))
 
  #it shound multiple thd to scale back to the original U and V valuse
  flow=flow*thd 
  print(flow.shape)
  
  plt.close()
  fig=plt.figure(figsize=(12,5))
  #physical latent space and change
  ax=fig.add_axes([0.05, 0.1, 0.35, 0.8])
  plotWSWDAxes(ax,'darkred','darkblue')
  diff=cntData[1,:,:]-cntData[0,:,:]
  plt.pcolormesh(xbins,ybins,diff,cmap='bwr',vmin=-35,vmax=35)
  #label the numbers on plot
  for (j,i),days in np.ndenumerate(diff):
    ax.text(0.5*(xbins[i]+xbins[i+1]),0.5*(ybins[j]+ybins[j+1]),'%d'%days,color='snow',ha='center',va='center')
  cb=plt.colorbar(extend='both')
  cb.set_label('Days',fontsize=12)
  plt.title('%s [SSP-585]-[Historical]\nSynoptic Flow Regime Change'%m,fontsize=12,loc='left')

  axes=[fig.add_axes([0.43, 0.1, 0.24, 0.8]),fig.add_axes([0.69, 0.1, 0.24, 0.8])]
  for i,ax in enumerate(axes):
    lbl='Generated Local Circulation\n$WD=%.1f^\circ WS=%.1f m/s$'%(wdList[i],wsList[i])

    ax.contour(lonTW,latTW,topo,levels=[0.01,],colors='k',linewidths=2)
    ax.contourf(lonTW,latTW,topo_m,cmap='Greys')
    generated_ws=np.sqrt(flow[i,0,:,:]**2+flow[i,1,:,:]**2)
    strm=ax.streamplot(xx,yy,flow[i,0,:,:],flow[i,1,:,:],
                        color=generated_ws,cmap='YlGnBu',norm=mc.Normalize(vmin=0.0,vmax=7.0),
                        linewidth=2,density=2.2,zorder=3,arrowsize=2.5)
    for c in cities:
        ax.plot(cities[c][0],cities[c][1],marker='*',markersize=20,mfc='yellow',mec='k')
        ax.text(cities[c][0],cities[c][1]-0.27,c,fontsize=15,fontweight='bold',ha='center')
    ax.set_xlim(119.8,122.2)
    ax.set_ylim(21.7,25.5)
    ax.set_title(lbl,fontsize=12)

    ax.set_xticks([])
    ax.set_yticks([])
  plt.tight_layout()
  ax_cb_ws = fig.add_axes([0.94, 0.1, 0.01, 0.8])
  cbar_ws=plt.colorbar(strm.lines,cax=ax_cb_ws,extend='max')
  cbar_ws.set_label('Wind Speed (m/s)',fontsize=12)

  #arrows
  style = "Simple, tail_width=2, head_width=8, head_length=10"
  kw = dict(arrowstyle=style, color="k")
  arrow1 = matplotlib.patches.FancyArrowPatch(
      (0.31, 0.41), (0.73, 0.15), transform=fig.transFigure,  
      connectionstyle="arc3,rad=0.5", **kw)
  fig.patches.append(arrow1)
  arrow2 = matplotlib.patches.FancyArrowPatch(
      (0.32, 0.86), (0.48, 0.84), transform=fig.transFigure,  
      connectionstyle="arc3,rad=-0.3", **kw)
  fig.patches.append(arrow2)


  #plt.suptitle('Various Generated Local Circulations Corresonding to Synoptic Flow Regime Change',fontsize=50)

  plt.annotate('(a)', xy=(0.02, 0.94), xytext=(0.02, 0.94),xycoords='figure fraction',fontsize=15)
  plt.annotate('(b)', xy=(0.42, 0.95), xytext=(0.42, 0.95),xycoords='figure fraction',fontsize=15)
  plt.annotate('(c)', xy=(0.68, 0.95), xytext=(0.68, 0.95),xycoords='figure fraction',fontsize=15)

  plt.savefig('./figures/fig8_local_circulation_shift.png',dpi=400)




