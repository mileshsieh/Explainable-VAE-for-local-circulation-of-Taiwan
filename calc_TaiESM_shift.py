from scipy.optimize import fsolve
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import dataUtils as du
import matplotlib.colors as mc
import matplotlib,glob
import matplotlib.ticker as ticker
matplotlib.rc('xtick',labelsize=12)
matplotlib.rc('ytick',labelsize=12)

mu_x=np.linspace(-4,2,100)
mu_y=np.linspace(-3,3,100)
grid_x,grid_y=np.meshgrid(mu_x,mu_y)

[a,b,x0,y0]=[0.0531645,0.89609678,10.06394176,-0.02435647]
[a0,a1,wd0]=[2.506213,-0.99329623,110-1.24772201]
def ws_h(x, y):
    return a * (x - x0)**2 + b * (y - y0)**2
def wd_h(x, y):
    #return theta[0]*(x-theta[2])**2 - theta[1]*y
    return a1*np.arctan(y/(x-a0))/np.pi*180+wd0

def plotWSWDAxes(plt,clr_WD,clr_WS):
    c_wd=plt.contour(grid_x,grid_y,wd_h(grid_x,grid_y),colors=[clr_WD])
    plt.gca().clabel(c_wd, c_wd.levels, inline=True, fmt='$WD=%.0f\degree$', fontsize=7)

    c_ws=plt.contour(grid_x,grid_y,ws_h(grid_x,grid_y),colors=[clr_WS])
    plt.gca().clabel(c_ws, c_ws.levels, inline=True, fmt='WS=%.1f m/s', fontsize=7)
    #plt.xticks([])
    #plt.yticks([])

    return plt
def xy2ws(x,y):
    return 0.0531645*(x-10.06394176)**2+0.89609678*(y+0.02435647)**2
#wind direction
def xy2wd(x,y):
    return -0.99329623*np.arctan(y/(x-2.506213))/np.pi*180-1.24772201+110
def wind2xy(wd,ws):
    #[ws,wd]=[7.884263646649078, 82.41238045081957]
    def equations(vars):
        x, y = vars
        #[a,b,x0,y0]=[0.0531645,0.89609678,10.06394176,-0.02435647]
        #[a0,a1,wd0]=[2.506213,-0.99329623,110-1.24772201]
        alpha=np.tan((wd-wd0)*np.pi/180/a1)
        eq1 = x-a0-y/alpha
        eq2 = a*(x-x0)**2+b*(y-y0)**2-ws
        return [eq1, eq2]

    x, y =  fsolve(equations, (2.1, 0))
    return x,y

def calcDensity(xy,xbins,ybins):
    #xtk=ytk=0.5*(bins[1:]+bins[:-1])
    xbin=np.digitize(xy[:,0],xbins)
    ybin=np.digitize(xy[:,1],ybins)
    cnt=np.zeros((ybins.shape[0]-1,xbins.shape[0]-1))

    for i in range(xbins.shape[0]-1):
        for j in range(ybins.shape[0]-1):
            #print(i,j,xbin[(xbin==i)&(ybin==j)].shape[0])
            cnt[j,i]=xbin[(xbin==i+1)&(ybin==j+1)].shape[0]
    return cnt

if __name__=='__main__':
    #load TaiESM synoptic flow
    runDict={'histroical':['Historical',1990,2010],
           'SSP-585':['SSP-585',2079,2099],
            }
    p=925
    xbins=np.linspace(-4,2,7)
    ybins=np.linspace(-3,3,7)
    col='wd%d'%p
    ws_col='ws%d'%p
    m='TaiESM1'
    dfDict={}
    cntList=[]
    for run in runDict.keys():
        df_all=pd.read_csv(glob.glob('./data/CMIP6/synopticWind.%s.%s.*.csv'%(m,run))[0],header=0,parse_dates=[0])
        df_all.dropna(how='any',inplace=True)
        df_all=df_all[(df_all.date>='%d-10-01'%runDict[run][1])&(df_all.date<='%d-04-30'%runDict[run][2])]

        df=df_all[(df_all.wd925>30)&(df_all.wd925<180)&(df_all.ws925>3)&(df_all.wdCoh<=90)]
        print(m,run,df.date.size,df_all.date.size)
        wsList=df.ws925.values
        wdList=df.wd925.values

        xy=np.array([wind2xy(wd,ws) for wd,ws in zip(wdList,wsList)])
        cnt=calcDensity(xy,xbins,ybins)
        df=df.assign(**{'latent_x': xy[:,0], 'latent_y': xy[:,1],'xbin':np.digitize(xy[:,0],xbins),'ybin':np.digitize(xy[:,1],ybins)})
        dfDict[run]=df
        cntList.append(cnt)
    runList=list(runDict.keys())
    nRun=len(runList)

    #plot the flow regime change between current climate and future
    plt.close()
    plt.figure(figsize=(16,5))
    for i in range(len(cntList)):
        run=runList[i]
        plt.subplot(1,3,i+1)
        plt=plotWSWDAxes(plt,'orangered','skyblue')
        plt.pcolormesh(xbins,ybins,cntList[i],vmin=0,vmax=150)
        plt.title('%s %s\n%d-%d\n(%d days)'%(m,runDict[run][0],runDict[run][1],runDict[run][2],cntList[i].sum()),fontsize=12,loc='left')
        cb=plt.colorbar(extend='max')
        cb.set_label('Days',fontsize=12)
    plt.subplot(133)
    plt=plotWSWDAxes(plt,'orangered','skyblue')
    plt.pcolormesh(xbins,ybins,cntList[1]-cntList[0],cmap='bwr',vmin=-50,vmax=50)
    cb=plt.colorbar(extend='both')
    cb.set_label('Days',fontsize=12)
    plt.title('%s [SSP-585]-[Historical]\nSynoptic Flow Regime Change'%m,fontsize=12,loc='left')
    plt.savefig('./figures/%s.flow_regimes_changes.png'%m)

    np.save('./data/CMIP6/synoptic_regime_days.%s.npy'%m,np.array(cntList))


