#!/home/mileshsieh/anaconda3/bin/python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import dataUtils as du
from numpy.linalg import norm
import matplotlib
import matplotlib.ticker as ticker
matplotlib.rc('xtick',labelsize=18)
matplotlib.rc('ytick',labelsize=18)

def speed_labels(bins, units):
    labels = []
    for left, right in zip(bins[:-1], bins[1:]):
        if left == bins[0]:
            labels.append('calm'.format(right))
        elif np.isinf(right):
            labels.append('>{} {}'.format(left, units))
        else:
            labels.append('{} - {} {}'.format(left, right, units))
        print(left,right,labels[-1])
    return list(labels)

if __name__=='__main__':
  dataset='ctrl'
  #load data
  caseList,X=du.load_dataset(dataset)
  #get synoptic factors
  features=['wd925', 'ws925']
  df=du.getSynoptic(caseList,features,dataset)
  ws=df.ws925.values
  wd=df.wd925.values
  print(ws.shape,wd.shape)

  #create ws bins
  spd_bins=[0,2,4,6,8,10,np.inf]
  ws_cnt,ws_bin_edges=np.histogram(ws,spd_bins)
  ws_bin_edges[-1]=12 #for xticks
  ws_x=ws_bin_edges[:-1]+1
  
  #create wd bins
  dir_bins=np.arange(-15, 225, 30)
  wd_cnt,wd_bin_edges=np.histogram(wd,dir_bins)
  wd_x=wd_bin_edges[:-1]+15

  plt.close()
  plt.figure(figsize=(20,10))
  ax1=plt.subplot(121)
  ax1.bar(wd_x,wd_cnt,width=15)
  plt.gca().set_xticks(wd_x)
  plt.gca().set_xticklabels(['%.0f'%x for x in wd_x])
  plt.xlim(wd_x[0]-15,wd_x[-1]+15)
  plt.ylim(0,85)
  plt.grid(True)
  plt.xlabel('Wind Direction ($^{\circ}$)',fontsize=20)
  plt.ylabel('Number of Cases',fontsize=20)

  ax2=plt.subplot(122)
  ax2.bar(ws_x,ws_cnt)
  wslabels=['%.0f-%.0f'%(x-1,x+1) for x in ws_x[:-1]]+['>=10']
  plt.gca().set_xticks(ws_x)
  plt.gca().set_xticklabels(wslabels)
  #plt.xlim(0.0,12.0)
  plt.ylim(0,85)
  plt.grid(True)
  plt.xlabel('Wind Speed (m/s)',fontsize=20)
  #plt.ylabel('Occurence',fontsize=20)

  plt.suptitle('Prescribed Synoptic Near-Surface Winds in Selected Cases',fontsize=30)
  plt.annotate('(a)', xy=(0.13, 0.85), xytext=(0.13, 0.85),xycoords='figure fraction',fontsize=20)
  plt.annotate('(b)', xy=(0.55, 0.85), xytext=(0.55, 0.85),xycoords='figure fraction',fontsize=20)
  plt.savefig('./figures/fig2_IC_distribution.png',dpi=300)


