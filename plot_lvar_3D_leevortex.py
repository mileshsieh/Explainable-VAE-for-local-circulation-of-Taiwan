#!/home/mileshsieh/anaconda3/bin/python
import numpy as np
import dataUtils as du
from matplotlib import pyplot as plt
import matplotlib.colors as mc
import matplotlib
import matplotlib.ticker as ticker

matplotlib.rc('xtick',labelsize=25)
matplotlib.rc('ytick',labelsize=25)

if __name__=='__main__':
  dataset='ctrl'
  #load the latent variables
  a=np.load('./data/VAE/latent/latent_var.vae61x61_ldim3_b0.0100_ctrl_t6to54_seed3_norm11.npy')
  print(a.shape)
  ncase,nt,_=a.shape

  #load the synoptic factors
  features=['wd925', 'ws925','LTS']
  caseList=du.getCaseList(dataset)
  df=du.getSynoptic(caseList,features,dataset)
  ws=df.ws925.values
  wd=df.wd925.values
  lts=df.LTS.values
  print(ws.shape,wd.shape)

  pltCfg={'WD':[60,180,'Wind Direction','Wind Direction($^{\circ}$)'],
          'WS':[2,10,'Wind Speed','Wind Speed(m/s)'],
          'LTS':[10,20,'LTS','LTS (K)'],
        }

  for var,varData in zip(['LTS',],[lts,]):
    plt.close()
    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=90, azim=90)
    for tt in range(nt):
      sc=ax.scatter(a[:,tt,0],a[:,tt,1],a[:,tt,2],cmap='jet',marker='o',
                    c=lts,vmin=pltCfg[var][0],vmax=pltCfg[var][1],alpha=0.3)
    cb=plt.colorbar(sc,extend='both')
    cb.set_label(pltCfg[var][3],fontsize=25)
    plt.grid(True)
    plt.title('3D Latent Space',fontsize=25)
  
    k=0
    for zz in range(0,75,3):
      k=k+1
      el=90-zz
      az=90
      ax.view_init(elev=el, azim=az)
      print(k,el,az)
      plt.savefig('./figures/VAE3D/VAE_3D.%s.%03d.png'%(var,k))
    for ii in range(0,360,3):
      k=k+1
      el=15
      az=90+ii
      ax.view_init(elev=el, azim=az)
      print(k,el,az)
      plt.savefig('./figures/VAE3D/VAE_3D.%s.%03d.png'%(var,k))
    for zz in range(0,75,3):
      k=k+1
      el=15+zz
      az=90
      ax.view_init(elev=el, azim=az)
      print(k,el,az)
      plt.savefig('./figures/VAE3D/VAE_3D.%s.%03d.png'%(var,k))
  

