import sys
sys.path.insert(1,'/home/jsperezc/jupyter/AQ_Forecast/functions/')
import Funciones_Lectura1 as FL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
import datetime as dt
import copy

## Read wind
fechaf=dt.datetime.strftime(dt.datetime.now(),'%Y%m%d %H:%M')
fechai=dt.datetime.strftime(dt.datetime.now()-dt.timedelta(hours = 24*5),'%Y%m%d %H:%M')

Omega, Zonal, Meridional, lv2 =FL.lee_RWP(fechai,fechaf,60)
h=lv2[lv2.index==lv2.index[0]].values[0]

## Read Fires
df_fires = pd.read_csv('/var/data1/AQ_Forecast_DATA/operational/Fires/MODIS_C6_1_South_America_7d.csv')
str_time = df_fires['acq_time'].values.astype(str)
str_time = np.array([str_time[i].zfill(4) for i in range(len(str_time))])
str_date = df_fires['acq_date'].values.astype(str)

dates_fires = np.array([dt.datetime.strptime(str_date[i]+' '+str_time[i],'%Y-%m-%d %H%M').replace(minute = 0) \
    for i in range(len(str_time))])
df_fires.index = dates_fires
df_fires.index = df_fires.index - dt.timedelta(hours = 5)

### Compute IFRP
U = copy.deepcopy(Zonal[15])
V = copy.deepcopy(Meridional[15])
Mag = np.sqrt(U**2+V**2)
Dir = np.mod(180+np.rad2deg(np.arctan2(U, V)),360)

angle = 135
distance = 1800
timeback = 48

U48 = copy.deepcopy(U.rolling(str(timeback)+'H').mean())
V48 = copy.deepcopy(V.rolling(str(timeback)+'H').mean())

Mag = np.sqrt(U48**2+U48**2)
Dir = pd.DataFrame(np.mod(180+np.rad2deg(np.arctan2(U48, V48)),360))

cone_angle = angle
Dir_Min = Dir - cone_angle/2.
Dir_Max = Dir + cone_angle/2.
Dir_Max[Dir_Max>360] = Dir_Max[Dir_Max>360] - 360
Dir_Min[Dir_Min<0] = Dir_Max[Dir_Max<0] + 360
Dir_Max.columns = ['dirmax']
Dir_Min.columns = ['dirmin']

lat_center = 6.25
lon_center = -75.25

def get_bearing(lat1, long1, lat2, long2):
    dLon = (long2 - long1)
    x = np.cos(np.radians(lat2)) * np.sin(np.radians(dLon))
    y = np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(dLon))
    brng = np.arctan2(x,y)
    brng = np.degrees(brng)

    return np.mod(brng,360)

def get_distance(lat1,lon1,lat2,lon2):
    R = 6373.0
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(np.radians(dlat/2)))**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * (np.sin(np.radians(dlon/2.)))**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    return distance

df_fires['bearing'] = get_bearing(lat_center,lon_center,df_fires.latitude.values,df_fires.longitude.values)
df_fires['distance'] = get_distance(lat_center,lon_center,df_fires.latitude.values,df_fires.longitude.values)

ifrp = []
for date_temp in U48.index:
    if (date_temp-dt.timedelta(hours = 1)) in Dir_Max.index:
        dirmax = Dir_Max.loc[date_temp-dt.timedelta(hours = 1)][0]
        dirmin = Dir_Min.loc[date_temp-dt.timedelta(hours = 1)][0]
        df_fires_temp = df_fires.loc[str(date_temp - dt.timedelta(hours = 48)):\
            str(date_temp)]
        df_fires_temp = df_fires_temp[df_fires_temp.distance<distance]

        if len(df_fires_temp)==0:
            ifrp.append(0)
        else:
            b = df_fires_temp.bearing.values

            if dirmin>dirmax:
                where = np.where((b>=dirmin)|(b<=dirmax))
            else:
                where = np.where((b>=dirmin)&(b<=dirmax))
            ifrp.append(df_fires_temp.iloc[where].frp.sum())
    else:
        ifrp.append(np.nan)
df = pd.DataFrame(ifrp,index = U48.index,columns = ['IFRP']).dropna()
df.to_csv('/var/data1/AQ_Forecast_DATA/operational/Fires/IFRP.csv')


### PLOT

import os
import matplotlib.pyplot as plt

from cartopy import config
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

path_shape = '/home/jsperezc/jupyter/AQ_Forecast/notebooks/Shapes/'
os.environ["CARTOPY_USER_BACKGROUNDS"] = '/home/jsperezc/jupyter/AQ_Forecast/operational_scripts/cartopy_bg/'

fig = plt.figure(figsize=(12, 9))

ax = plt.axes(projection=ccrs.PlateCarree())
plt.title('Incendios detectados por MODIS (6.1) en las últimas 48 horas\n('+
          str(df.index[-1]-dt.timedelta(hours = 48))+' a '+str(df.index[-1])+')',fontsize = 16)

ax.coastlines(resolution='50m', color='white', linewidth=0.8)
ax.set_extent([-80, -65, 2, 13], ccrs.PlateCarree())
ax.background_img(name='BM', resolution='high')

shapefile = list(shpreader.Reader(path_shape+'Shape_AMVA').geometries())
ax.add_geometries(shapefile, ccrs.PlateCarree(),edgecolor='white',facecolor='none', linewidth=0.8, zorder=4)
shapefile = list(shpreader.Reader(path_shape+'Departamentos').geometries())
ax.add_geometries(shapefile, ccrs.PlateCarree(),edgecolor='white',facecolor='none', linewidth=0.4)


ax.scatter(df_fires_temp.longitude,df_fires_temp.latitude,color='orange',alpha = 0.1)
ax.scatter(df_fires_temp.iloc[where].longitude,df_fires_temp.iloc[where].latitude,alpha = 0.3,color='red',
          label = 'Puntos dentro del cono de influencia\nPotencia radiativa de incendios\nintegrada: '+\
                str(int(df.iloc[-1][0]))+'MW')
plt.legend(loc='upper left',fontsize=16)

path_fig = '/home/jsperezc/jupyter/AQ_Forecast/operational_results/figures/mean/archive/IFRP_'+\
    str(df.index[-1]).replace(':','').replace(' ','_').replace('-','')+'.png'
# plt.show()
plt.savefig(path_fig,bbox_inches='tight')
os.system('cp '+path_fig+' '+path_fig.replace(str(df.index[-1]).replace(':','').\
                                              replace(' ','_').replace('-',''),'Current').replace('/archive',''))

path_1 = path_fig
path_2 = path_fig.replace(str(df.index[-1]).replace(':','').\
                                              replace(' ','_').replace('-',''),'Current').replace('/archive','')

### Copia a SAL:

local_path = '/home/jsperezc/jupyter/AQ_Forecast/operational_results/'
sal_path = '/var/www/jhayron/AQ_Forecast/operational_results/'


os.system('scp '+path_1+' jsperezc@siata.gov.co:'+path_1.replace(local_path,sal_path))
os.system('scp '+path_2+' jsperezc@siata.gov.co:'+path_2.replace(local_path,sal_path))

