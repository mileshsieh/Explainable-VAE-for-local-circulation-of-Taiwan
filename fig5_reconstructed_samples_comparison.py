#!/home/mileshsieh/anaconda3/bin/python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import dataUtils as du
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.colors as mc
import matplotlib
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.ticker as ticker
matplotlib.rc('xtick',labelsize=25)
matplotlib.rc('ytick',labelsize=25)

def plotStreamLine(ax,topo,vardata,title,rtitle,ltitle):
    ny,nx=topo.shape
    xx,yy=np.meshgrid(np.arange(nx),np.arange(ny))
    ax.contour(topo,levels=[0.05,],colors=['k'])
    #ax.contour(topo,levels=[0.05,0.2],colors=['k'])
    ws=np.sqrt(vardata[0,:,:]**2+vardata[1,:,:]**2)
    #print(ws.min(),ws.max())
    strm=ax.streamplot(xx,yy,vardata[0,:,:],vardata[1,:,:],
            color=ws,cmap='YlGnBu',norm=mc.Normalize(vmin=0.0,vmax=7.0),linewidth=2,density=0.8,zorder=3,arrowsize=1.5)
    #ax.plot([xstart,xend,xend,xstart,xstart],[ystart,ystart,yend,yend,ystart],lw=2,ls='--')
    ax.contourf(topo,levels=[0.2,5.0],colors=['darkgreen'],zorder=5)
    ax.set_xlim(0,nx)
    ax.set_ylim(0,ny)
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.set_title(rtitle,fontsize=10,loc='right')
    ax.set_title(ltitle,fontsize=10,loc='left')
    ax.set_title(title,fontsize=18,loc='center')
    return strm

if __name__=='__main__':
  num_input_channels=2
  latent_dim=2
  beta=0.01
  dataset='ctrl'
  thd=11.71
  seed=3
 
  suffix={'VAE':'vae61x61_ldim%d_b%.4f_%s_t%dto%d_seed%d_norm%d'%(latent_dim,beta,dataset,du.ts,du.te,seed,int(thd)),
          'PCA':'PCA.2pcs',
          }
  #load data
  caseList,X=du.load_dataset(dataset)
  nt,ncase,nvar,ny,nx=X.shape
  print(X.shape)

  recon_cases=['ish20100110s_chem','ish20070409s_chem','ish20111227s_chem',\
               'ish20161001s_chem','ish20101129s_chem','ish20110430s_chem']
  #select 7th frame of the testing samples in these specific experiments as the reconstruction snapshots
  itt=6
  data={}
  for m in ['Input','VAE','PCA']:
    data[m]=[]

  idx_cases=[np.where(caseList==c)[0][0] for c in recon_cases]
  #get the indices of test dataset
  testing_indices=np.load('./data/VAE/input/testing_indices.npy')
  idx_test_cases,idx_test_tt=np.divmod(testing_indices,nt)
  for idx in idx_cases:
    print(caseList[idx])
    tmp=sorted(idx_test_tt[idx_test_cases==idx])
    data['Input'].append(X[tmp[itt],idx,:,:,:])
    print('get input data',idx,tmp[itt],X[tmp[itt],idx,:,:,:].shape)
    for m in ['VAE','PCA']:
        recon=np.load('./data/VAE/reconstruction/recon.%s.%s.npy'%(caseList[idx],suffix[m]))
        print('get %s recon data'%m,idx,tmp[itt],recon.shape)
        data[m].append(recon[tmp[itt],:,:,:])

  step=5
  topo=du.load_topo(du.ys,du.ye,du.xs,du.xe,step=step)

  nmodel=3
  nsample=6
  plt.close()
  #fig=plt.figure(figsize=(20,16),layout='constrained')
  fig=plt.figure(figsize=(20,12))
  grid=AxesGrid(fig,111,nrows_ncols=(nmodel,nsample),axes_pad=(0.1,0.3),share_all=True)

  irow=0
  title_list=['a','b','c','d','e','f']
  for irow,m in enumerate(['Input','PCA','VAE']):
    for icol,idx in enumerate(idx_cases):
      print(m,caseList[idx_cases[icol]])

      #if m=='Input':
      #  strm2=plotStreamLine(grid[irow,icol],topo,data[m][icol],'%s-%d'%(title_list[ititle],irow+1),m,'')
      #  ititle=ititle+1
      #else:
      strm2=plotStreamLine(grid[irow*nsample+icol],topo,data[m][icol],'%s-%d'%(title_list[icol],irow+1),'','')
      if icol==0:
        grid[irow*nsample+icol].set_ylabel(m,fontsize=30)
    irow=irow+1
  #plt.suptitle(pltCfg[p_option][0],fontsize=50)
  #plt.text(0.06,0.35,'Input',fontsize=35,ha='center',transform=fig.transFigure)
  #plt.text(0.06,0.15,'Output',fontsize=35,ha='center',transform=fig.transFigure)

  plt.tight_layout()
  fig.subplots_adjust(right=0.9)
  ax_cb = fig.add_axes([0.92, 0.14, 0.03, 0.70])
  cbar=plt.colorbar(strm2.lines,cax=ax_cb,extend='max')
  cbar.set_label('Wind Speed (m/s)',fontsize=30)
  plt.suptitle('Intercomparison of reconstructed local circulations using PCA/VAE',fontsize=35)
  plt.savefig('./figures/fig5_reconstruction_samples.png',dpi=300)
  
  
