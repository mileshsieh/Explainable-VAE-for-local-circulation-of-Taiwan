#!/home/mileshsieh/anaconda3/bin/python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#pca
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import dataUtils as du
import matplotlib.colors as mc
import matplotlib
import matplotlib.ticker as ticker
matplotlib.rc('xtick',labelsize=15)
matplotlib.rc('ytick',labelsize=15)

def plotStreamLine(ax,topo,vardata,title,rtitle,ltitle,step=5):
    ny,nx=topo.shape
    xx,yy=np.meshgrid(np.arange(nx),np.arange(ny))
    ax.contour(topo,levels=[0.05,],colors=['k'])
    #ax.contour(topo,levels=[0.05,0.2],colors=['k'])
    ws=np.sqrt(vardata[0,:,:]**2+vardata[1,:,:]**2)
    print(ws.min(),ws.max())
    strm=ax.streamplot(xx[::step,::step],yy[::step,::step],vardata[0,::step,::step],vardata[1,::step,::step],
            color=ws[::step,::step],cmap='viridis',norm=mc.Normalize(vmin=0.0,vmax=15.0),density=1.4,zorder=3)
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
  thd=11.71
  #load data
  dataset='ctrl'
  #load data
  topo=np.load('./data/VAE/input/topo.npy')
  caseList,X=du.load_dataset(dataset)
  print(X.shape)

  #scale
  nt,ncase,nvar,ny,nx=X.shape
  X=(X/thd).reshape(nt*ncase,nvar*ny*nx)

  #get the indices of test dataset
  testing_indices=np.load('./data/VAE/input/testing_indices.npy')
  training_indices=np.load('./data/VAE/input/training_indices.npy')

  idx_test_cases,idx_test_tt=np.divmod(testing_indices,nt)
  idx_train_cases,idx_train_tt=np.divmod(training_indices,nt)
  test_cases=[]
  train_cases=[]
  for tt in range(nt):
    test_cases.append(sorted(idx_test_cases[idx_test_tt==tt]))
    train_cases.append(sorted(idx_train_cases[idx_train_tt==tt]))

  X_input=X[training_indices]
  X_test=X[testing_indices]
  nsample=len(X_input)
  print('PCA input data:')
  print('nsample=%d nvar=%d ny=%d nx=%d'%(nsample,nvar,ny,nx))

  #pca
  npc=2
  pca=PCA(n_components=npc).fit(X_input)
  print('PCA: npc=%d'%npc)

  #plot explaination
  print('PCA explained ratio:',pca.explained_variance_ratio_.sum())
  plt.close()
  plt.bar(np.arange(npc)+1,pca.explained_variance_ratio_)
  plt.xticks(np.arange(npc)+1)
  plt.title('%.3f%% with %d PCs'%(pca.explained_variance_ratio_.sum()*100,npc),fontsize=20)
  plt.savefig('./figures/PCA/exp_ratio.%d.png'%npc)

  #plot modes
  #modes=np.array([scaler.inverse_transform(pca.components_[k,:]).reshape(nvar,ny,nx) for k in range(npc)])
  modes=np.array([pca.components_[k,:].reshape(nvar,ny,nx) for k in range(npc)])
  np.save('./data/VAE/PCA/modes.%d.npy'%npc,modes)
  
  for k in range(npc):
    print('plot PC%d'%(k+1))
    plt.close()
    fig=plt.figure(figsize=(16,16))
    ax1=plt.subplot(111)
    strm1=plotStreamLine(ax1,topo,modes[k,:,:,:],'PC%d'%(k+1),'','')
    #fig.subplots_adjust(right=0.85)
    #ax_cb = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    #cbar=plt.colorbar(strm1.lines,cax=ax_cb)
    #cbar.set_label('Wind Speed (m/s)',fontsize=20)
    plt.savefig('./figures/PCA/PCs.%dof%d.png'%(k+1,npc))
  

  #get coef of first npc compoment
  coef=X.dot(pca.components_.T)

  #reconstruction
  recon=coef.dot(pca.components_).reshape(nt,ncase,nvar,ny,nx)*thd
  for icase,case in enumerate(caseList):
    np.save('./data/VAE/reconstruction/recon.%s.PCA.%dpcs.npy'%(case,npc),recon[:,icase,:,:])

  coef=np.swapaxes(coef.reshape(nt,ncase,npc),0,1)
  print(coef.shape)
  np.save('./data/VAE/PCA/coef.%d.npy'%npc,coef)

  #scatterplot of PCA coef
  #get synoptic factors
  features=['wd925', 'ws925']
  df=du.getSynoptic(caseList,features,dataset)
  ws=df.ws925.values
  wd=df.wd925.values
  print(ws.shape,wd.shape)

  pltCfg={'WD':['Wind Direction (deg)','WD(deg)',60,180,'Accent'],
          'WS':['Wind Speed (m/s)','WS(m/s)',2,10,'Dark2'],
         }
  for lbl,var in zip(['WD','WS'],[wd,ws]):
    plt.close()
    plt.figure(figsize=(16,16))
    for tt in range(nt):
      plt.scatter(coef[train_cases[tt],tt,0],coef[train_cases[tt],tt,1],c=var[train_cases[tt]],s=50,marker='o',vmin=pltCfg[lbl][2],vmax=pltCfg[lbl][3],cmap=pltCfg[lbl][4])
      plt.scatter(coef[test_cases[tt],tt,0],coef[test_cases[tt],tt,1],c=var[test_cases[tt]],s=50,marker='x',vmin=pltCfg[lbl][2],vmax=pltCfg[lbl][3],cmap=pltCfg[lbl][4])
    cb=plt.colorbar()
    cb.set_label(pltCfg[lbl][1],fontsize=35)
    plt.grid(True)
    plt.title(pltCfg[lbl][0],fontsize=35)
    plt.xlabel('PC1',fontsize=35)
    plt.ylabel('PC2',fontsize=35)
    plt.savefig('figures/latent_features.%s.%s.png'%(lbl,'PCA'))

