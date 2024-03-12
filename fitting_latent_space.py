#!/home/mileshsieh/anaconda3/bin/python

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import dataUtils as du
from scipy.optimize import fsolve
import matplotlib
import matplotlib.colors as mc
import matplotlib.ticker as ticker
from scipy.interpolate import griddata
from scipy.optimize import least_squares

matplotlib.rc('xtick',labelsize=15)
matplotlib.rc('ytick',labelsize=15)

if __name__=='__main__':
  latent_dim=2
  beta=0.01
  dataset='ctrl'
  thd=11.71
  seed=3
  #for write out
  sf='vae61x61_ldim%d_b%.4f_%s_t%dto%d_seed%d_norm%d'%(latent_dim,beta,dataset,du.ts,du.te,seed,int(thd))

  #load the latent variables
  latent_var=np.load('data/VAE/latent/latent_var.%s.npy'%sf)
  print(latent_var.shape)
  nCase,nt,_=latent_var.shape
  #load the synoptic factors
  features=['wd925', 'ws925']
  caseList=du.getCaseList(dataset)
  df=du.getSynoptic(caseList,features,dataset)
  ws=df.ws925.values
  wd=df.wd925.values

  # training by tt=6to60 (08:00 to 02:00) so that tt=0 means 08:00
  pltCfg={'WD':['Wind Direction (deg)','WD(deg)'],
          'WS':['Wind Speed (m/s)','WS(m/s)'],
        }

  # linear interpolation of x in [-2,2] and y in [-2,2] latent space
  mu_x=np.linspace(-2,2,100)
  mu_y=np.linspace(-2,2,100)
  muxx,muyy=np.meshgrid(mu_x,mu_y)
  wd_mu=np.tile(wd,(48,1)).T
  grid_wd=griddata(latent_var.reshape((nCase*nt,-1)),wd_mu.ravel() , (muxx, muyy), method='linear')
  ws_mu=np.tile(ws,(48,1)).T
  grid_ws=griddata(latent_var.reshape((nCase*nt,-1)),ws_mu.ravel() , (muxx, muyy), method='linear')

  #fitting function form of ws
  def h_ws(theta, x, y):
    return theta[0] * (x - theta[2])**2 + theta[1] * (y - theta[3])**2

  xx,yy=np.meshgrid(mu_x,mu_y)
  def fun_ws(theta):
    return (h_ws(theta, xx, yy) - grid_ws).flatten()

  #initial guess of parameters of h_ws
  a = 1.0
  b = 1.0
  x0 = 0.0
  y0 = 0.0
  theta0 = [a, b, x0, y0]
  res_ws = least_squares(fun_ws, theta0)
  ws_fitting=h_ws(res_ws.x, xx, yy)
  print('fitting params of ws:',res_ws.x)
  
  #fitting function form of wd
  def h_wd(theta, x, y):
    return theta[0]*np.arctan(y/(x-theta[2]))/np.pi*180+theta[1]

  def fun_wd(theta):
    return (h_wd(theta, xx, yy) - (grid_wd)).flatten()

  #initial guess of parameters of h_wd
  a = 1.0
  b = 1.0
  x0 = 2.0
  theta0 = [a,b,x0]
  res_wd = least_squares(fun_wd, theta0)  
  wd_fitting=h_wd(res_wd.x, xx, yy)
  print('fitting params of wd:',res_wd.x)
  
  #calculate mean error
  print('Mean Error of fitting WS:',np.mean(np.abs(ws_fitting-grid_ws)))
  print('Mean Error of fitting WD:',np.mean(np.abs(wd_fitting-grid_wd)))
  
  #plot
  plt.close()
  fig=plt.figure(figsize=(14,6))
  ax1=plt.subplot(121)
  c1=ax1.contourf(xx,yy,grid_wd)
  ax1.set_xticks([])
  ax1.set_yticks([])
  cb=plt.colorbar(c1)
  cb.set_label('Wind Direction (deg)',fontsize=15)
  c_wd=ax1.contour(xx,yy,wd_fitting,colors=['lightcyan'])
  ax1.clabel(c_wd, c_wd.levels, inline=True, fmt='$WD=%.0f\degree$', fontsize=10)

  ax2=plt.subplot(122)
  c2=ax2.contourf(xx,yy,grid_ws)
  ax2.set_xticks([])
  ax2.set_yticks([])
  cb=plt.colorbar(c2)
  cb.set_label('Wind Speed (m/s)',fontsize=15)
  c_ws=ax2.contour(xx,yy,ws_fitting,colors=['lightcyan'])
  ax2.clabel(c_ws, c_ws.levels, inline=True, fmt='WS=%.1f m/s', fontsize=10)


