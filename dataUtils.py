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
  #setup data directory
  dataDir='/home/mileshsieh/leeVortex/data'
  #load data
  caseList=getCaseList(dataset)
  topo=np.load('%s/topodata.npy'%dataDir)[ystart:yend,xstart:xend]
  uall=[]
  vall=[]
  for icase in range(len(caseList)):
    print(caseList[icase])
    case=caseList[icase]
    if scaled:
      u,u_scaler = remove_topo_wind(np.load('%s/u_data.%s.npy'%(dataDir,case))[tstart:tend,ystart:yend,xstart:xend],topo)
      v,v_scaler = remove_topo_wind(np.load('%s/v_data.%s.npy'%(dataDir,case))[tstart:tend,ystart:yend,xstart:xend],topo)
    else:
      u = remove_topo_wind(np.load('%s/u_data.%s.npy'%(dataDir,case))[tstart:tend,ystart:yend,xstart:xend],topo,scaled=False)
      v = remove_topo_wind(np.load('%s/v_data.%s.npy'%(dataDir,case))[tstart:tend,ystart:yend,xstart:xend],topo,scaled=False)
    
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

def load_dataset(dataset):
    caseList=getCaseList(dataset)
    data=[]
    for c in caseList:
        data.append(np.load('data/VAE/input/%s_dataset.%s.npy'%(dataset,c)))
    result=np.swapaxes(np.array(data),0,1)
    return caseList,result


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

if __name__=='__main__':
  #create_case_dir()
  #load data
  for dataset in ['training','testing']:
    X,topo,caseList=load_leevortex_data(ts,te,ys,ye,xs,xe,dataset,scaled=False,reshape=False,step=5)
    print(dataset,X.shape)
    for i,c in enumerate(caseList):
      np.save('data/VAE/input/%s_dataset.%s.npy'%(dataset,c),X[:,i,:,:,:])
  np.save('data/VAE/input/topo.npy',topo)
  #scale
  #nt,ncase,nvar,ny,nx=X.shape
  #scaler=StandardScaler()
  #nX=scaler.fit_transform(X.reshape(nt*ncase,nvar*ny*nx)).reshape(nt*ncase,nvar,ny,nx)
  #print(X.shape)
