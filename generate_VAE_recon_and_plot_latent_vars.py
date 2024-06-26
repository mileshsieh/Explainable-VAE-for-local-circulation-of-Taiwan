#!/home/mileshsieh/anaconda3/bin/python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from train_VAE_uv2uv_61x61 import variationalAutoEncoder,Encoder,Decoder,Reshape
import glob,random,os
import dataUtils as du
from itertools import permutations
import matplotlib.colors as mc
import matplotlib
import matplotlib.ticker as ticker
matplotlib.rc('xtick',labelsize=25)
matplotlib.rc('ytick',labelsize=25)


if __name__=='__main__':
  num_input_channels=2
  latent_dim=2
  beta=0.01
  #for activation function
  dataset='ctrl'
  thd=11.71

  seed=3
  #for write out
  sf='vae61x61_ldim%d_b%.4f_%s_t%dto%d_seed%d_norm%d'%(latent_dim,beta,dataset,du.ts,du.te,seed,int(thd))
  device = torch.device('cpu')

  # Load model
  model_ae = torch.load('data/VAE/model/leevortex.%s.pth'%sf,map_location=torch.device('cpu'))
  model_ae.eval()

  #load data
  caseList,X=du.load_dataset(dataset)
  print(X.shape)

  #scale
  nt,ncase,nvar,ny,nx=X.shape
  X=(X/thd)

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

  print(X.shape,testing_indices[:10])

  latent_mu_all=[]
  for icase,case in enumerate(caseList):
    input_data=torch.from_numpy(X[:,icase,:,:,:]).float()  

    # reconstruction
    with torch.no_grad():
      inputs = input_data
      #output of vae is (recon, mu, logvar)
      outputs,mu,logvar = model_ae(inputs.to(device))
    output_X = outputs.detach().cpu().numpy()
    output_X=output_X*thd
    np.save('data/VAE/reconstruction/recon.%s.%s.npy'%(case,sf),output_X)

    latent_mu=mu.detach().cpu().numpy()
    latent_mu_all.append(latent_mu)
    print(case,latent_mu.shape)
    
   
  latent_mu_all=np.array(latent_mu_all)
  print(latent_mu_all.shape)
  np.save('data/VAE/latent/latent_var.%s.npy'%sf,latent_mu_all)
  
  #get synoptic factors
  features=['wd925', 'ws925']
  df=du.getSynoptic(caseList,features,dataset)
  ws=df.ws925.values
  wd=df.wd925.values
  print(ws.shape,wd.shape)

  # training by tt=6to60 (08:00 to 02:00) so that tt=0 means 08:00
  pltCfg={'WD':['Wind Direction (deg)','WD(deg)',60,180,'Accent'],
          'WS':['Wind Speed (m/s)','WS(m/s)',2,10,'Dark2'],
         }
  for lbl,var in zip(['WD','WS'],[wd,ws]):
    
    #plot each 2D relationship in latent space
    for (v1,v2) in permutations(np.arange(latent_mu_all.shape[2]),2):
      plt.close()
      plt.figure(figsize=(16,16))
      for tt in range(nt):
        plt.scatter(latent_mu_all[train_cases[tt],tt,v1],latent_mu_all[train_cases[tt],tt,v2],c=var[train_cases[tt]],marker='o',vmin=pltCfg[lbl][2],vmax=pltCfg[lbl][3],cmap=pltCfg[lbl][4])
        plt.scatter(latent_mu_all[test_cases[tt],tt,v1],latent_mu_all[test_cases[tt],tt,v2],c=var[test_cases[tt]],s=50,marker='x',vmin=pltCfg[lbl][2],vmax=pltCfg[lbl][3],cmap=pltCfg[lbl][4])
      cb=plt.colorbar()
      cb.set_label(pltCfg[lbl][1],fontsize=35)
      plt.grid(True)
      plt.title(pltCfg[lbl][0],fontsize=35)
      plt.xlabel('var[%d]'%v1,fontsize=35)
      plt.ylabel('var[%d]'%v2,fontsize=35)
      plt.savefig('figures/latent_features.%s.%s.%dx%d.png'%(lbl,sf,v1,v2))
