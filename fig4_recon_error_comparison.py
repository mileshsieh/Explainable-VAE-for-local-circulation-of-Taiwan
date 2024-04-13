#!/home/mileshsieh/anaconda3/bin/python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import dataUtils as du
from numpy.linalg import norm
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.colors as mc
import matplotlib
import matplotlib.ticker as ticker
matplotlib.rc('xtick',labelsize=18)
matplotlib.rc('ytick',labelsize=18)

if __name__=='__main__':
  num_input_channels=2
  latent_dim=2
  beta=0.01
  dataset='ctrl'
  thd=11.71
  seed=3
 
  suffix={'VAE':'vae64x64_ldim%d_b%.4f_%s_t%dto%d_seed%d_norm%d'%(latent_dim,beta,dataset,du.ts,du.te,seed,int(thd)),
          'PCA':'PCA.2pcs',
          }
  mList=['PCA','VAE']
  #load data
  caseList,X=du.load_dataset(dataset)
  nt,ncase,nvar,ny,nx=X.shape
  X=X.reshape(nt*ncase,nvar,ny,nx)
  print('Input',X.shape)

  #pile up reconstruction
  X_recon={}
  for m in mList:
    tmp=[]
    for c in caseList:
      recon=np.load('./data/VAE/reconstruction/recon.%s.%s.npy'%(c,suffix[m]))
      #print('get %s recon data'%m,c,recon.shape)
      tmp.append(recon)
    X_recon[m]=np.swapaxes(np.array(tmp),0,1).reshape(nt*ncase,nvar,ny,nx)
    print(m,X_recon[m].shape)

  #split dataset
  #get the indices of test dataset
  testing_indices=np.load('./data/VAE/input/testing_indices.npy')
  training_indices=np.load('./data/VAE/input/training_indices.npy')

  mse={'training':[],'testing':[]}
  for label,idxList in zip(['training','testing'],[training_indices,testing_indices]):
    for m in mList:
      err=np.array([(X_recon[m][k,:,:,:]-X[k,:,:,:])**2 for k in idxList]).mean()
      mse[label].append(err)

  
  #plot mse
  x=np.arange(len(mList))  # the label locations
  width=0.3  # the width of the bars
  mul=-1
  plt.close()
  fig,ax=plt.subplots(figsize=(12,8))
  for d,er in mse.items():
    offset=width*mul
    print(x,offset)
    rects=ax.bar(x+width*mul,er,width,align='edge',label=d)
    ax.bar_label(rects,fmt='%.2f',padding=3,fontsize=15)
    mul+=1

  # Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_ylabel('Mean Square Error (m/s)',fontsize=20)
  ax.set_title('Reconstruction Errors',fontsize=20)
  ax.set_xticks(x, mList)
  ax.legend(loc='upper right',fontsize=15)
  ax.set_ylim(0, 3.2)

  plt.savefig('./figures/fig4_reconstruction_errors.jpg',format='jpg',dpi=300)
