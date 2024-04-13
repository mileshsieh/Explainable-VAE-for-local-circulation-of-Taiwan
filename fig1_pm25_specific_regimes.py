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

#windrose citiation:
#https://gist.github.com/phobson/41b41bdd157a2bcf6e14
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

def pm25_labels(bins, units):   
    labels = []
    for left, right in zip(bins[:-1], bins[1:]):
        if left == bins[0]:
            labels.append('<{} {}'.format(right, units))
        elif np.isinf(right):
            labels.append('>{} {}'.format(left, units))
        else:
            labels.append('{} - {} {}'.format(left, right, units))

    return list(labels)


def _convert_dir(directions, N=None):
    if N is None:
        N = directions.shape[0]
    barDir = directions * np.pi/180. - np.pi/N
    barWidth = 2 * np.pi / N
    return barDir, barWidth

def wind_rose(rosedata, wind_dirs, ax=None,palette=None):
    if palette is None:
        #palette = seaborn.color_palette("Spectral", n_colors=rosedata.shape[1])
        palette = seaborn.color_palette("inferno", n_colors=rosedata.shape[1])

    bar_dir, bar_width = _convert_dir(wind_dirs)

    if ax==None:
        fig, ax = plt.subplots(figsize=(16, 16), subplot_kw=dict(polar=True))
    ax.set_theta_direction('clockwise')
    ax.set_theta_zero_location('N')

    for n, (c1, c2) in enumerate(zip(rosedata.columns[:-1], rosedata.columns[1:])):
        print(n,c1,c2)
        if n == 0:
            # first column only
            ax.bar(bar_dir, rosedata[c1].values, 
                   width=bar_width,
                   color=palette[0],
                   edgecolor='none',
                   label=c1,
                   linewidth=0)

        # all other columns
        ax.bar(bar_dir, rosedata[c2].values, 
               width=bar_width, 
               bottom=rosedata.cumsum(axis=1)[c1].values,
               color=palette[n+1],
               edgecolor='none',
               label=c2,
               linewidth=0)

    #xtl = ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

    #ax.xaxis.set_major_locator(FixedLocator(np.array([20.,50.,80.,110.,140.,170.0,200.,230.,260.,290.,320.,350.,])*np.pi/180.0))
    ax.xaxis.set_major_locator(FixedLocator(np.linspace(0,330,12)*np.pi/180.0))

    # For the minor ticks, use no labels; default NullFormatter.
    ax.xaxis.set_minor_locator(FixedLocator(np.array([0.,90.,180.,270.])*np.pi/180.0))
    ax.xaxis.set_minor_formatter(FixedFormatter(['N','E','S','W']))
    ax.xaxis.grid(True, which='minor')
    ax.tick_params(axis='x', which='minor', labelsize=30)
    ax.grid(which='minor', axis='x',linewidth=3,color='k')
    ax.set_ylim(0,200)
    ax.yaxis.set_ticks([0,50,100,150])
    #ax.set_thetamin(30)
    #ax.set_thetamax(170)
    return ax

if __name__=='__main__':
    cities={'Taipei':[121.5598,25.09108],'Kaohsiung':[120.311922,22.620856]}
    
    df=pd.read_csv('./data/obs/noWx.obs.47918_wind.2008to2019.csv',header=0,usecols=[0,4,5],parse_dates=[0])
    df['date']=df.apply(lambda ser: ser.yyyymmdd.strftime('%Y%m%d'),axis=1)
    #df=df[(df.yyyymmdd>='2008-01-01')&(df.yyyymmdd<='2019-12-31')]
    
    def getFlowRegimeData(df,wd,ws):
        return df[(df.wd925>=wd-15)&(df.wd925<=wd+15)&(df.ws925>=ws-1)&(df.ws925>=ws+1)]
        
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
    
    
    #for windrose
    df.dropna(how='any',inplace=True)
    #create ws bins    
    spd_bins = [-1,0,2,4,6,8,10,np.inf]
    spd_labels = speed_labels(spd_bins, units='m/s')
    spd_dict={}
    for i,lbl in enumerate(spd_labels):
        spd_dict['%d'%i]=lbl
    #create wd bins
    dir_bins = np.arange(-15, 380, 30)
    dir_labels = (dir_bins[:-1] + dir_bins[1:]) / 2    
    
    #first: all years
    df_for_bin=df[['yyyymmdd','wd925','ws925']]
    p=925
    col='wd%d'%p
    ws_col='ws%d'%p
    
    plt.close()
    fig=plt.figure(figsize=(24,8))
    ax=plt.subplot(1,3,1,polar=True)
    tplt='Synoptic Flow Regimes [2008-2019]\n(%d Local Circulation-dominated Days)'
    total_count=df_for_bin.shape[0]
    calm_count = df_for_bin.query("%s==0"%ws_col).shape[0]
    
    print('Of {} total observations, {} have calm winds.'.format(total_count, calm_count))
    print(len(dir_labels),len(dir_bins))
    print(len(spd_labels),len(spd_bins))
    wscut = pd.cut(df_for_bin[ws_col], bins=spd_bins, labels=list(spd_dict.keys()), right=True)
    wdcut = pd.cut(df_for_bin[col], bins=dir_bins, labels=dir_labels, right=False)
    df_bin = df_for_bin.assign(
            WindSpd_bins=lambda df: pd.cut(df[ws_col], bins=spd_bins, labels=list(spd_dict.keys()), right=True))\
            .assign(WindDir_bins=lambda df:pd.cut(df[col], bins=dir_bins, labels=dir_labels, right=False))\
            .replace({'WindDir_bins': {0: 360}})
        
    size_2D=df_bin.groupby(by=['WindSpd_bins', 'WindDir_bins']).size()
    rose=(size_2D.reindex(pd.MultiIndex.from_product([wscut.cat.categories, wdcut.cat.categories]))
          .unstack()
          .fillna(0.0)
          .sort_index(axis=1)).T
          #.applymap(lambda x: x / total_count * 100)).T
    rose.drop([0.0],inplace=True)
    rose.rename(columns=spd_dict,inplace=True)
    #print(rose.index.unique())
    
    #directions = np.arange(0, 360, 15)
    directions = dir_labels[1:]+15
    #ax=plt.subplot(1,2,iplt+1,projection='polar')
    ax = wind_rose(rose, directions,ax)
    ax.plot([np.pi*0.59,np.pi*0.59,np.pi*0.74,np.pi*0.74,np.pi*0.59],[67,124,124,67,67],lw=3,color='b')
    ax.annotate('WD120',(np.pi*0.59*0.95,75),color='b',fontsize=20)
    ax.plot([np.pi*0.75,np.pi*0.75,np.pi*0.917,np.pi*0.917,np.pi*0.75],[50,80,80,50,50],lw=3,color='darkgreen')
    ax.annotate('WD150',(np.pi*0.917*1.2,100),color='darkgreen',fontsize=20)
    leg = ax.legend(loc=(-0.25, -0.2), ncol=3,fontsize=20)
    
    ax.set_title(tplt%total_count,fontsize=30)
    
    #fig=plt.figure(figsize=(16,10))
    i=1
    wdList=[120,150]
    ws=7.0
    for wd in wdList:
        frData=getFlowRegimeData(df,wd,ws)
        ddList=frData.date.values.astype(int)
        print('Regime WS%.0f WD%.0d: mean wind wd=%.5f, ws=%.5f'%(ws,wd,frData.wd925.mean(),frData.ws925.mean()))
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
            fig.add_axes([0.45,0.05,0.2,0.8])
        else:
            fig.add_axes([0.7,0.05,0.2,0.8])
        plt.contour(lonTW,latTW,topo,levels=[0.01,],colors='k',linewidths=2)
        plt.contourf(lonTW,latTW,topo_m,cmap='Greys')
        for c in cities:
            plt.plot(cities[c][0],cities[c][1],marker='*',markersize=30,mfc='yellow',mec='k')
            plt.text(cities[c][0],cities[c][1]-0.3,c,fontsize=25,fontweight='bold',ha='center')
        cs=plt.scatter(lon,lat,c=enh_idx,s=50,vmax=0.8,vmin=0.1,cmap='jet')
        plt.xlim(119.8,122.2)
        plt.ylim(21.7,25.5)
        plt.title('WD%d\n'%wd+lbl,fontsize=25)        
        plt.xticks([])
        plt.yticks([])
        i=i+1
    #fig.subplots_adjust(right=0.9)
    ax_cb = fig.add_axes([0.92, 0.05, 0.01, 0.8])
    cbar=plt.colorbar(cs,cax=ax_cb, extend='both', ticks=np.linspace(0.1,0.8,8))
    cbar.set_label('$PM_{2.5}$ Enhancement Index',fontsize=20)
    
    plt.annotate('(a)', xy=(0.1, 0.9), xytext=(0.03, 0.93),xycoords='figure fraction',fontsize=30)
    plt.annotate('(b)', xy=(0.4, 0.9), xytext=(0.45, 0.93),xycoords='figure fraction',fontsize=30)
    plt.annotate('(c)', xy=(0.7, 0.9), xytext=(0.7, 0.93),xycoords='figure fraction',fontsize=30)
 
    
    plt.savefig('./figures/fig1_WD125_WD150_enhancement_index.jpg',format='jpg',dpi=300)
