import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import dataUtils as du
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.colors as mc
import matplotlib
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
    ax.set_title(title,fontsize=20,loc='center')
    return strm

if __name__=='__main__':
  num_input_channels=2
  latent_dim=2
  beta=0.1
  dataset='ctrl'
  thd=11
  seed=3
 
  suffix={'VAE':'vae61x61_ldim%d_b%.4f_%s_t%dto%d_seed%d_norm%d'%(latent_dim,beta,dataset,du.ts,du.te,seed,thd),
          'CAE':'cae61x61_ldim%d_%s_t%dto%d_seed%d_norm%d'%(latent_dim,dataset,du.ts,du.te,seed,thd),
          'PCA':'PCA.2pcs',
          }
  #load data
  caseList,X=du.load_dataset(dataset)
  nt,ncase,nvar,ny,nx=X.shape
  print(X.shape)
  recon_cases=['ish20100110s_chem','ish20070409s_chem','ish20111227s_chem',\
               'ish20161001s_chem','ish20101129s_chem','ish20110430s_chem']
  #select 7th frame as the reconstruction snapshots
  itt=6
  data={}
  for m in ['Input','VAE','CAE','PCA']:
    data[m]=[]

  idx_cases=[np.where(caseList==c)[0][0] for c in recon_cases]
  #get the indices of test dataset
  testing_indices=np.load('./data/AE/input/testing_indices.npy')
  idx_test_cases,idx_test_tt=np.divmod(testing_indices,nt)
  for idx in idx_cases:
    print(caseList[idx])
    tmp=sorted(idx_test_tt[idx_test_cases==idx])
    data['Input'].append(X[tmp[itt],idx,:,:,:])
    print('get input data',idx,tmp[itt],X[tmp[itt],idx,:,:,:].shape)
    for m in ['VAE','CAE','PCA']:
        recon=np.load('./data/AE/reconstruction/recon.%s.%s.npy'%(caseList[idx],suffix[m]))
        print('get %s recon data'%m,idx,tmp[itt],recon.shape)
        data[m].append(recon[tmp[itt],:,:,:])

  step=5
  topo=du.load_topo(du.ys,du.ye,du.xs,du.xe,step=step)

  plt.close()
  fig, axes = plt.subplots(ncols=6,nrows=4,constrained_layout=False,figsize=(20,12))
  irow=0
  title_list=['a','b','c','d','e','f']
  for irow,m in enumerate(['Input','VAE','CAE','PCA']):
    ititle=0
    for iax,ax in enumerate(axes[irow,:]):
      print(m,caseList[idx_cases[iax]])

      if m=='input':
        strm2=plotStreamLine(ax,topo,data[m][iax],title_list[ititle],'','')
        ititle=ititle+1
      else:
        strm2=plotStreamLine(ax,topo,data[m][iax],'','','')

    irow=irow+1
  plt.tight_layout()
  #plt.suptitle(pltCfg[p_option][0],fontsize=50)
  #plt.text(0.06,0.35,'Input',fontsize=35,ha='center',transform=fig.transFigure)
  #plt.text(0.06,0.15,'Output',fontsize=35,ha='center',transform=fig.transFigure)

  fig.subplots_adjust(right=0.88,left=0.12)
  ax_cb = fig.add_axes([0.9, 0.03, 0.03, 0.8])
  cbar=plt.colorbar(strm2.lines,cax=ax_cb,extend='max')
  cbar.set_label('Wind Speed (m/s)',fontsize=30)
  '''
  err=np.load('data/VAE/epo_err.%s.npy'%sf)
  ax_err=fig.add_axes([0.2,0.62,0.6,0.3])
  ax_err.plot(np.arange(err.shape[0])*10,err,lw=4)
  plt.xlim(0,200)
  plt.ylim(0.001,0.0025)
  plt.yticks(np.linspace(0.001,0.0025,4))
  plt.grid()
  ax_err.set_title('Loss vs. Training Epochs',fontsize=35)
  plt.xlabel('Epochs',fontsize=30)
  plt.ylabel('Loss',fontsize=30)

  #annotate on the latent space
  plt.annotate('(a)', xy=(0.15, 0.41), xytext=(0.05, 0.95),xycoords='figure fraction',fontsize=40)
  plt.annotate('(b)', xy=(0.25, 0.51), xytext=(0.05, 0.5),xycoords='figure fraction',fontsize=40)
  plt.savefig('./figures/fig3_loss_recon_comp.png',dpi=400)
  '''
