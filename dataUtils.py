#!/home/mileshsieh/anaconda3/bin/python
import numpy as np
import pandas as pd
import glob,os
from sklearn.preprocessing import StandardScaler

def remove_topo_wind(udata,topo,scaled=True):
  nt,ny,nx=udata.shape
  u=np.array([np.where(topo>1,0.0,udata[k,:,:]) for k in range(nt)]).reshape((nt,ny*nx))
  if scaled:
    scaler=StandardScaler()
    u_std=scaler.fit_transform(u).reshape((nt,ny,nx))
    return u_std,scaler
  else:
    return u.reshape((nt,ny,nx))

def create_case_dir():
  caseList=getCaseList()
  for case in caseList:
    os.system('mkdir %s'%case)
  return

def getCaseList(dataset):
  #fList=sorted(glob.glob('./data/u_data.ish*.npy'))
  #caseList=[a.split('.')[-2] for a in fList]
  a=pd.read_csv('data/VAE/synoptic.%s.csv'%dataset,header=0,parse_dates=[0])
  caseList=a.caseName.values
  return caseList

def load_topo(ystart,yend,xstart,xend,step=0):
  topo=np.load('data/topodata.npy')[ystart:yend,xstart:xend]
  if step>0:
      topo=topo[::step,::step]
  return topo

def load_leevortex_data(tstart,tend,ystart,yend,xstart,xend,dataset,scaled=True,reshape=True,step=0):
  #load data
  caseList=getCaseList(dataset)
  topo=np.load('data/topodata.npy')[ystart:yend,xstart:xend]
  uall=[]
  vall=[]
  for icase in range(len(caseList)):
    print(caseList[icase])
    case=caseList[icase]
    if scaled:
      u,u_scaler = remove_topo_wind(np.load('data/VAE/trainingDataset/u_data.%s.npy'%case)[tstart:tend,ystart:yend,xstart:xend],topo)
      v,v_scaler = remove_topo_wind(np.load('data/VAE/trainingDataset/v_data.%s.npy'%case)[tstart:tend,ystart:yend,xstart:xend],topo)
    else:
      u = remove_topo_wind(np.load('data/VAE/trainingDataset/u_data.%s.npy'%case)[tstart:tend,ystart:yend,xstart:xend],topo,scaled=False)
      v = remove_topo_wind(np.load('data/VAE/trainingDataset/v_data.%s.npy'%case)[tstart:tend,ystart:yend,xstart:xend],topo,scaled=False)
    
    uall.append(u)
    vall.append(v)
  u=np.array(uall)
  v=np.array(vall)
  X=np.swapaxes(np.array([u,v]),0,2)
  #now shape is (nt,ncase,nvar,ny,nx)
  nt,ncase,nvar,ny,nx=X.shape
  print(X.shape)
  if step>0:
      X=X[:,:,:,::step,::step]
      topo=topo[::step,::step]
  if reshape:
    return X.reshape(nt*ncase,nvar,ny,nx),topo,caseList
  else:
    return X,topo,caseList

def getSynoptic(caseList,features,dataset):
    #parser = lambda date: pd.datetime.strptime(date, '%Y-%m-%d')
    a=pd.read_csv('data/VAE/synoptic.%s.csv'%dataset,header=0,parse_dates=[0])
    #a=pd.read_csv('inversion_SE.csv',header=0,parse_dates=[0,],date_parser=parser)
    #a['case']=a.apply(lambda ser:'ish%ds_chem'%ser.date,axis=1)
    a.set_index('caseName',inplace=True)

    return a.loc[caseList,features]

#for lee vortex domain
#(2,300,300)
ts=6
te=54
ys=150
ye=450
xs=20
xe=320
step=0
#cut (2,301,301) and downsampling into (2,61,61) by step=5
ys=150
ye=451
xs=20
xe=321

#skip tt between 0,47
# skip 10 samples
skip_tt=[10, 44, 13, 28,  3, 35, 31, 25, 43, 14]
#skip 16 samples (1/3 of 48 snapshots)
#from numpy.random import default_rng
#rng = default_rng()
#numbers = rng.choice(48, size=16, replace=False)
skip_tt=[33, 21,  1, 42,  0, 34, 38, 15, 40, 43, 32, 36, 29, 14, 46, 39]

#step=5 is set in the VAE script

if __name__=='__main__':
  #create_case_dir()
  #load data
  X,topo,caseList=load_leevortex_data(ts,te,ys,ye,xs,xe,scaled=False,reshape=False,step=5)
  print(X.shape)

  #scale
  #nt,ncase,nvar,ny,nx=X.shape
  #scaler=StandardScaler()
  #nX=scaler.fit_transform(X.reshape(nt*ncase,nvar*ny*nx)).reshape(nt*ncase,nvar,ny,nx)
  #print(X.shape)
