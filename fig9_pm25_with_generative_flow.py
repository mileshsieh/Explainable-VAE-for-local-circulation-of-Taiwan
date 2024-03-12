#!/home/mileshsieh/anaconda3/bin/python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mc
import seaborn
import matplotlib
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter,FixedLocator,FixedFormatter
matplotlib.rc('xtick',labelsize=20)
matplotlib.rc('ytick',labelsize=20)

def getDateList(df,wd,ws):
    df_within=df[(df.wd925>=wd-15)&(df.wd925<=wd+15)&(df.ws925>=ws-1)&(df.ws925>=ws+1)]
    dlist=df_within.date.values.astype(int)
    return dlist

if __name__=='__main__':
    cities={'Taipei':[121.5598,25.09108],'Kaohsiung':[120.311922,22.620856]}
    df=pd.read_csv('./data/obs/noWx.obs.47918_wind.csv',header=0,usecols=[0,12,13],parse_dates=[0])
    df['date']=df.apply(lambda ser: ser.yyyymmdd.strftime('%Y%m%d'),axis=1)
    df=df[(df.yyyymmdd>='2008-01-01')&(df.yyyymmdd<='2019-12-31')]

    xbins=np.linspace(-4,2,13)
    ybins=np.linspace(-3,3,13)    
    
    pm25r_data=pd.read_csv('./data/obs/PM25ratio_74sta_20082019ColdSeason.csv').values
    lon=pm25r_data[0,1:].astype(float)
    lat=pm25r_data[1,1:].astype(float)
    #remove east stations
    idx_west=~((lon>121)&(lat<24.8))
    lon=lon[idx_west]
    lat=lat[idx_west]
    pm25r=pm25r_data[2:,1:].astype(float)[:,idx_west]
    dateList=pm25r_data[2:,0].astype(int)
    
    #tw flow and topo
    ys=150
    ye=451
    xs=20
    xe=321
    lonTW=np.load('./data/tw_s_lon.npy')[xs:xe]
    latTW=np.load('./data/tw_s_lat.npy')[ys:ye]
    topo=np.load('./data/topodata.npy')[ys:ye,xs:xe]
    topo_m=np.copy(topo)
    topo_m[topo_m==0]=np.nan

    xx,yy=np.meshgrid(lonTW[::5],latTW[::5])

    #local circulation + PM25
    flow={'WD120':np.load('./data/VAE/generation/flow_WD117.35_WS9.32.npy'),
          'WD150':np.load('./data/VAE/generation/flow_WD148.07_WS9.32.npy')
          }
    plt.close()
    fig=plt.figure(figsize=(16,10))
    i=1
    wdList=[120,150]
    ws=7.0
    for wd in wdList:
        regime='WD%d'%wd
        ddList=getDateList(df,wd,ws)
        lbl='$WD=%d\pm15^\circ WS=%.0f\pm1 m/s$'%(wd,ws)

        idxList=np.array([np.where(dateList==dd)[0][0] for dd in ddList])
        pm25r_days=pm25r[idxList,:]
    
        enh_idx=np.zeros(lon.shape[0])
        for k in range(pm25r_days.shape[1]):
            stn_pm25r=pm25r_days[:,k]
            stn_pm25r=stn_pm25r[~np.isnan(stn_pm25r)]
            enh=(np.where(stn_pm25r>1.0,1,0).sum())/stn_pm25r.shape[0]
            #print(k,enh,np.where(stn_pm25r>1.0,1,0).sum(),stn_pm25r.shape[0])
            enh_idx[k]=enh
        if i==1:
            fig.add_axes([0.05,0.05,0.35,0.8])
        else:
            fig.add_axes([0.42,0.05,0.35,0.8])
        plt.contour(lonTW,latTW,topo,levels=[0.01,],colors='k',linewidths=2)
        plt.contourf(lonTW,latTW,topo_m,cmap='Greys')
        for c in cities:
            plt.plot(cities[c][0],cities[c][1],marker='*',markersize=20,mfc='yellow',mec='k')
            plt.text(cities[c][0],cities[c][1]-0.27,c,fontsize=20,fontweight='bold',ha='center')
        cs=plt.scatter(lon,lat,c=enh_idx,s=50,vmax=0.8,vmin=0.1,cmap='jet')
        generated_ws=np.sqrt(flow[regime][0,:,:]**2+flow[regime][1,:,:]**2)
        strm=plt.streamplot(xx,yy,flow[regime][0,:,:],flow[regime][1,:,:],
                            color=generated_ws,cmap='YlGnBu',norm=mc.Normalize(vmin=0.0,vmax=7.0),linewidth=2,density=2.2,zorder=3,arrowsize=2.5)
        plt.xlim(119.8,122.2)
        plt.ylim(21.7,25.5)
        #plt.title('WD%d\n'%wd+lbl,fontsize=25)
        plt.title('WD%d'%wd,fontsize=25)
        
        plt.xticks([])
        plt.yticks([])
        i=i+1

    ax_cb = fig.add_axes([0.79, 0.05, 0.03, 0.8])
    cbar=plt.colorbar(cs,cax=ax_cb, extend='both',ticks=np.linspace(0.1,0.8,8))
    cbar.set_label('$PM_{2.5}$ Enhancement Index',fontsize=20)
    ax_cb_ws = fig.add_axes([0.9, 0.05, 0.03, 0.8])
    cbar_ws=plt.colorbar(strm.lines,cax=ax_cb_ws,extend='max')
    cbar_ws.set_label('Wind Speed (m/s)',fontsize=20)

    plt.suptitle('Generated Local Circulation using VAE',fontsize=30)
    plt.annotate('(a)', xy=(0.04, 0.7), xytext=(0.035, 0.88),xycoords='figure fraction',fontsize=30)
    plt.annotate('(b)', xy=(0.68, 0.7), xytext=(0.435, 0.88),xycoords='figure fraction',fontsize=30)
    #plt.annotate('(c)', xy=(0.7, 0.9), xytext=(0.63, 0.9),xycoords='figure fraction',fontsize=25)
 

    plt.savefig('figures/fig9_pm25_with_flow.png',dpi=300)
