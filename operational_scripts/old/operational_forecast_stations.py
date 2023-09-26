#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('/home/calidadaire/Paquete/')

import airquality.read_data as read

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib
import netCDF4 as nc
import datetime as dt
import copy
import statsmodels.api as sm

import sys
sys.path.insert(1,'/home/jsperezc/jupyter/AQ_Forecast/functions/')
import postprocessing
import forecasting
import xarray as xr
import sklearn
import glob
import os
import pickle
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import keras


# In[2]:


stations = ["ITA-CJUS", "CAL-LASA", "ITA-CONC", "MED-LAYE", "CAL-JOAR", "EST-HOSP", "MED-ALTA", "MED-VILL", 
            "BAR-TORR", "COP-CVID", "MED-BEME", "MED-TESO", "MED-SCRI", "MED-ARAN", "BEL-FEVE", "ENV-HOSP", 
            "SAB-RAME", "MED-SELE","CEN-TRAF"]


# In[3]:


datetime_now = dt.datetime.now()
forecast_initial_date = datetime_now.replace(minute = 0,second=0,microsecond=0)# - dt.timedelta(hours=1)
gfs_path = '/var/data1/AQ_Forecast_DATA/operational/GFS/'
cams_path = '/var/data1/AQ_Forecast_DATA/operational/CAMS/'
dic_forecasts_stations = {}
dic_pm_stations = {}

for station in stations:
    print(station)
    try:
        coor_esta= pd.read_csv("/home/jsperezc/jupyter/AQ_Forecast/notebooks/ForecastDevelopment/"+                "CoordenadasEstaciones.csv", index_col= "Nombre")
        lat_est = coor_esta.loc[station].Latitud
        lon_est = coor_esta.loc[station].Longitud


        pm2p5 = forecasting.get_pm2p5_for_forecast(datetime_now,station_name = station)

        df_CAMS,df_optimal_cams = forecasting.get_cams_for_forecast(forecast_initial_date,cams_path,operational = True,
                                                                   latlon = (lat_est,lon_est))
        df_GFS,df_optimal_gfs = forecasting.get_gfs_for_forecast(forecast_initial_date,gfs_path,operational = True,
                                                                latlon = (lat_est,lon_est))

        forecasts = pd.DataFrame(index = df_CAMS['pm2p5_cams'][forecast_initial_date:forecast_initial_date+                                                          dt.timedelta(hours = 95)].index)

        model_names = ['LR','MLP','RF','SVR','KNN','CAMS']

        for model_name in model_names:
            if model_name=='CAMS':
                vals_cams = df_CAMS['pm2p5_cams'][forecast_initial_date:forecast_initial_date+                                                         dt.timedelta(hours = 95)].values
#                 for i in range(0,96):
#                     folder_mos = '/var/data1/AQ_Forecast_DATA/trained_models/final_models_stations/mos_operators/'+                        station+'/'
#                     with open(folder_mos+model_name+'lt_'+str(i+1).zfill(2)+'.mos','rb') as f:
#                         mos_temp = pickle.load(f)
#                     vals_cams[i] = mos_temp(vals_cams[i])
                forecasts[model_name] = vals_cams
            else:
#                 forecasts[model_name] = forecasting.forecast_pm2p5_v2(forecast_initial_date,pm2p5,df_optimal_cams,                    df_optimal_gfs,model_name=model_name,station_name = station,mos_correction=True)
                forecasts[model_name] = forecasting.forecast_pm2p5_v4(forecast_initial_date,pm2p5,df_optimal_cams,                    df_optimal_gfs,model_name=model_name,station_name = station,mos_correction=False)
        forecasts['MEAN'] = np.mean(forecasts.iloc[:,:],axis = 1)
        forecasts['MIN'] = np.min(forecasts.iloc[:,:-1],axis = 1)
        forecasts['MAX'] = np.max(forecasts.iloc[:,:-2],axis = 1)
        dic_forecasts_stations[station] = forecasts
        dic_pm_stations[station] = pm2p5
        
        ### Guardar tabla:
        try:os.mkdir('/home/jsperezc/jupyter/AQ_Forecast/operational_results/tables/stations/archive/'+station+'/')
        except: pass
        path_table = '/home/jsperezc/jupyter/AQ_Forecast/operational_results/tables/stations/archive/'+            station+'/'+'Fc_'+station+'_'+            str(forecast_initial_date).replace(':','').replace(' ','_').replace('-','')+'.csv'
        forecasts.to_csv(path_table)
    except Exception as e: print(e)


# In[4]:


def plot_forecasts_individual_operational(name_station,forecast_initial_date,forecasts,pm2p5):
    ############# PLOT 1 ################

    fig = plt.figure(figsize=(11,5))

    dates_forecast = forecasts.index
    for model_name in forecasts.keys()[:-4]:
        if model_name == 'NN':
            plt.plot(dates_forecast,forecasts[model_name].values,label = 'MLP',ls='--',alpha=0.4)
        else:
            plt.plot(dates_forecast,forecasts[model_name].values,label = model_name,ls='--',alpha=0.4)

    plt.plot(dates_forecast,forecasts['CAMS'].values,label = 'CAMS (ECMWF)',ls='-.',alpha=0.5,color='gray')
    plt.plot(dates_forecast,np.mean(forecasts.iloc[:,:-3],axis = 1),label = 'Media del ensamble',
             color='teal',lw=1.4)
    plt.fill_between(dates_forecast,np.min(forecasts.iloc[:,:-2],axis = 1),np.max(forecasts.iloc[:,:-2],axis = 1),
                    alpha=0.1,color='skyblue',label='Rango del ensamble')
    plt.axvline(forecast_initial_date,color='k',ls = '--',label='Fecha inicial de pronóstico')
    plt.plot(pm2p5[forecast_initial_date-dt.timedelta(hours=12):],lw=1.4,color='k',label='Observaciones')

    plt.legend(ncol=4,bbox_to_anchor=(0.98, -0.12),fontsize=13)
    plt.xticks(fontsize=(12))
    plt.yticks(fontsize=(12))
    plt.ylabel('Concentración de PM2.5\n[$\mu g/m^3$]',fontsize=14)
    plt.xlim(dates_forecast[0]-dt.timedelta(hours=12),dates_forecast[-1])
    plt.grid(ls='--',alpha=0.3)
    plt.title(name_station+' - Fecha inicial del pronóstico: '+str(forecast_initial_date)+ ' HL',fontsize=15,loc='left',y = 1.02)

    ## TABLE
    table = forecasts[['MEAN','MIN','MAX']].iloc[:18]
    table = table.round(2)
    table['date'] = table.index.to_pydatetime().astype(str)
    table['date'] = [table['date'].iloc[i][:-3] for i in range(len(table))]

    table = table[['date','MEAN','MIN','MAX']]
    cell_text = []
    for row in range(len(table)):
        cell_text.append(table.iloc[row])

    table.columns = ['Fecha','Media [$\mu g / m^3$]','Mínimo [$\mu g / m^3$]','Máximo [$\mu g / m^3$]']
    ptable = plt.table(cellText=cell_text, colLabels=table.columns, bbox=(1,0,1,1),edges='vertical',
                      colLoc = 'center',rowLoc='center')
    ptable.auto_set_font_size(False)
    ptable.set_fontsize(12)
    ptable.auto_set_column_width((-1, 0, 1, 2, 3))

    for (row, col), cell in ptable.get_celld().items():
        if (row == 0):
            cell.set_text_props(fontproperties=FontProperties(weight='bold',size=12))

    text = plt.figtext(0.94, 0.91, 'Pronóstico de las próximas 18 horas',fontsize=15)
    t2 = ("* Este producto indica el pronóstico de la concentración promedio horaria de PM2.5 para\n"
          "las próximas 96 horas en las estaciones poblacionales del Valle de Aburrá.\n\n"
          "* Los modelos se desarrollaron usando información de aerosoles proveniente de CAMS (ECMWF) y\n"
          "meteorológica proveniente de GFS (NCEP). Cada línea punteada indica el pronóstico de un método\n"
          "estadístico distinto (LR, KNN, GB, SVR, MLP, RF, ANN), y su dispersión es proporcional a la\n"
          "incertidumbre asociada.\n\n")
    text2 = plt.figtext(0.94, -0.25, t2,fontsize=11, wrap=True)

    path_fig = '/home/jsperezc/jupyter/AQ_Forecast/operational_results/figures/stations/Fc_'+name_station+'_'+        'current.png'
    plt.savefig(path_fig,bbox_inches='tight')
    plt.close('all')


# In[5]:


def plot_forecasts_24h_individual_operational(name_station,forecast_initial_date,forecasts,pm2p5):
    ### FIGURA ICA:
    forecast_24h_mean = copy.deepcopy(forecasts)
    for model_name in forecasts.keys():
        forecast_24h_mean[model_name] = pd.concat([pm2p5['PM2.5'],forecasts[model_name]],axis=0).            rolling(24).mean()[forecasts.index]
    pm2p5_24h_mean = pm2p5.rolling(24).mean()

    limites_ica = {'C.A. Buena':(0,12.5),
                  'C.A. Moderada':(12.5,37.5),
                  'C.A. Dañina a grupos sensibles':(37.5,55.5),
                  'C.A. Dañina a la salud':(55.5,150.5),
                  'C.A. Muy dañina a la salud':(150.5,250.5),
                  'C.A. Peligrosa':(250.5,500)}

    colores_ica = ['green','yellow','orange','red','purple','brown']

    fig = plt.figure(figsize=(11,5))
    # fig.subplots_adjust(left=0.1, wspace=0.1)
    # plt.subplot2grid((1, 16), (0, 0), colspan=12)

    ## GRAPH

    dates_forecast = forecast_24h_mean.index
    for model_name in forecast_24h_mean.keys()[:-4]:
        if model_name == 'NN':
            plt.plot(dates_forecast,forecast_24h_mean[model_name].values,label = 'MLP',ls='--',alpha=0.4)
        else:
            plt.plot(dates_forecast,forecast_24h_mean[model_name].values,label = model_name,ls='--',alpha=0.4)

    plt.plot(dates_forecast,forecast_24h_mean['CAMS'].values,label = 'CAMS Corregido (ECMWF)',ls='-.',alpha=0.5,color='gray')
    plt.plot(dates_forecast,np.mean(forecast_24h_mean.iloc[:,:-4],axis = 1),label = 'Media del ensamble',
             color='teal',lw=1.4)
    plt.fill_between(dates_forecast,np.min(forecast_24h_mean.iloc[:,:-4],axis = 1),np.max(forecast_24h_mean.iloc[:,:-2],axis = 1),
                    alpha=0.35,color='skyblue',label='Rango del ensamble')
    plt.axvline(forecast_initial_date,color='k',ls = '--',label='Fecha inicial de pronóstico')
    plt.plot(pm2p5_24h_mean,lw=1.4,color='k',label='Observaciones')

    plt.xticks(fontsize=(12))
    plt.yticks(fontsize=(12))
    plt.ylabel('Concentración de PM2.5\n(promedios de 24 horas)\n[$\mu g/m^3$]',fontsize=14)
    plt.xlim(dates_forecast[0]-dt.timedelta(hours=48),dates_forecast[-1])
    plt.grid(ls='--',alpha=0.3)
    plt.title(name_station+' - Fecha inicial de pronóstico '+str(forecast_initial_date)+ ' HL',              fontsize=15,loc='left',y = 1.02)

    #### Background ICA

    for i in range(len(colores_ica)):
        plt.fill_between([dates_forecast[0]-dt.timedelta(hours=48),dates_forecast[-1]],
                    limites_ica[list(limites_ica.keys())[i]][0],
                    limites_ica[list(limites_ica.keys())[i]][1],
                    alpha=0.1,color=colores_ica[i],label=list(limites_ica.keys())[i])

    plt.ylim(np.min([pd.concat([pm2p5_24h_mean,forecast_24h_mean]).min().min()+10,0]),            np.max([pd.concat([pm2p5_24h_mean,forecast_24h_mean]).max().max()+10,0]))

    plt.legend(ncol=3,bbox_to_anchor=(0.98, -0.12),fontsize=13)

    ## TABLE

    # table_subplot = plt.subplot2grid((1, 16), (0, 15))
    # plt.axis('off')
    table = forecast_24h_mean[['MEAN','MIN','MAX']].iloc[:18]
    table = table.round(2)
    table['date'] = table.index.to_pydatetime().astype(str)
    table['date'] = [table['date'].iloc[i][:-3] for i in range(len(table))]

    table = table[['date','MEAN','MIN','MAX']]
    cell_text = []
    for row in range(len(table)):
        cell_text.append(table.iloc[row])

    table.columns = ['Fecha','Promedio [$\mu g / m^3$]','Mínimo [$\mu g / m^3$]','Máximo [$\mu g / m^3$]']
    ptable = plt.table(cellText=cell_text, colLabels=table.columns, bbox=(1,0,1,1),edges='vertical',
                      colLoc = 'center',rowLoc='center')
    ptable.auto_set_font_size(False)
    ptable.set_fontsize(12)
    ptable.auto_set_column_width((-1, 0, 1, 2, 3))

    for (row, col), cell in ptable.get_celld().items():
        if (row == 0):
            cell.set_text_props(fontproperties=FontProperties(weight='bold',size=12))

    text = plt.figtext(0.94, 0.91, 'Pronóstico de las próximas 18 horas',fontsize=15)
    t2 = ("* Este producto indica el pronóstico de la concentración de PM2.5 en promedios de 24 horas para\n"
          "las próximas 96 horas en las estaciones poblacionales del Valle de Aburrá.\n\n"
          "* Los modelos se desarrollaron usando información de aerosoles proveniente de CAMS (ECMWF) y\n"
          "meteorológica proveniente de GFS (NCEP). Cada línea punteada indica el pronóstico de un método\n"
          "estadístico distinto (LR, KNN, GB, SVR, MLP, RF, ANN), y su dispersión es proporcional a la\n"
          "incertidumbre asociada.\n\n")
    text2 = plt.figtext(0.94, -0.25, t2,fontsize=11, wrap=True)
    path_fig = '/home/jsperezc/jupyter/AQ_Forecast/operational_results/figures/stations/Fc24h_'+name_station+'_'+        'current.png'
    plt.savefig(path_fig,bbox_inches='tight')
    plt.close('all')


# In[6]:


for station in dic_forecasts_stations.keys():
    plot_forecasts_individual_operational(station,forecast_initial_date,                                          dic_forecasts_stations[station],dic_pm_stations[station])
    plot_forecasts_24h_individual_operational(station,forecast_initial_date,                                          dic_forecasts_stations[station],dic_pm_stations[station])


# In[11]:


dates_forecast = dic_forecasts_stations[station].index.to_pydatetime()


# In[12]:


## FIGURA 3 RESUMEN

cmaps = {'Norte':matplotlib.cm.get_cmap('Purples'),
        'Centro':matplotlib.cm.get_cmap('Greens'),
        'Sur':matplotlib.cm.get_cmap('Blues'),
        'Tráfico':matplotlib.cm.get_cmap('Oranges')}

colores_regiones = {'Norte':'purple',
        'Centro':'green',
        'Sur':'blue',
        'Tráfico':'orange'}


regiones_estaciones = {'Norte':np.array(['BAR-TORR','COP-CVID','BEL-FEVE','MED-VILL']),        'Centro':np.array(['MED-ARAN','MED-SCRI','MED-SELE','MED-BEME','MED-ALTA','MED-TESO']),        'Sur':np.array(['MED-LAYE','ITA-CJUS','ENV-HOSP','ITA-CONC','EST-HOSP','CAL-JOAR','CAL-LASA','SAB-RAME']),        'Tráfico':np.array(['CEN-TRAF','SUR-TRAF'])}

fig = plt.figure(figsize=(11,5))

tables = {}
for region in regiones_estaciones.keys():
    tables[region] = pd.DataFrame()

for i_station,station in enumerate(list(dic_forecasts_stations.keys())):
    if station in regiones_estaciones['Sur']:
        region = 'Sur'
    elif station in regiones_estaciones['Norte']:
        region = 'Norte'
    elif station in regiones_estaciones['Centro']:
        region = 'Centro'
    elif station in regiones_estaciones['Tráfico']:
        region = 'Tráfico'
        
    cmap = cmaps[region]
    where_station = np.where(regiones_estaciones[region]==station)[0][0]
    plt.plot(dic_forecasts_stations[station]['MEAN'],        color=cmap((((where_station+1)/(len(regiones_estaciones[region])*2))/0.8)+0.1),        ls = '--',alpha = 0.7)
    plt.plot(dic_pm_stations[station],        color=cmap((((where_station+1)/(len(regiones_estaciones[region])*2))/0.8)+0.1),        alpha = 0.7,label = station)
    
    tables[region][station] = dic_forecasts_stations[station]['MEAN']
    
plt.axvline(forecast_initial_date,color='k',ls = '--',label='Fecha inicial de pronóstico')
plt.xticks(fontsize=(12))
plt.yticks(fontsize=(12))
plt.ylabel('Concentración de PM2.5\n[$\mu g/m^3$]',fontsize=14)
plt.xlim(dates_forecast[0]-dt.timedelta(hours=12),dates_forecast[-1])
plt.grid(ls='--',alpha=0.3)
plt.title('Fecha inicial del pronóstico: '+str(forecast_initial_date)+ ' HL',fontsize=15,loc='left',y = 1.02)

table = pd.DataFrame()
for region in regiones_estaciones.keys():
    table[region] = tables[region].mean(axis=1)
    
    plt.plot(table[region],        color=colores_regiones[region],        alpha = 0.7,label = region)

plt.legend(ncol=4,bbox_to_anchor=(0.98, -0.12),fontsize=13)
table = table.iloc[:18]
table = table.round(2)
table['date'] = table.index.to_pydatetime().astype(str)
table['date'] = [table['date'].iloc[i][:-3] for i in range(len(table))]

table = table[['date','Norte','Centro','Sur']]
cell_text = []
for row in range(len(table)):
    cell_text.append(table.iloc[row])

table.columns = ['Fecha','Norte [$\mu g / m^3$]','Centro [$\mu g / m^3$]','Sur [$\mu g / m^3$]']
ptable = plt.table(cellText=cell_text, colLabels=table.columns, bbox=(1,0,1,1),edges='vertical',
                  colLoc = 'center',rowLoc='center')
ptable.auto_set_font_size(False)
ptable.set_fontsize(12)
ptable.auto_set_column_width((-1, 0, 1, 2, 3))

for (row, col), cell in ptable.get_celld().items():
    if (row == 0):
        cell.set_text_props(fontproperties=FontProperties(weight='bold',size=12))

text = plt.figtext(0.94, 0.91, 'Pronóstico de las próximas 18 horas en promedio en cada sub-región.',fontsize=15)
t2 = ("* Este producto indica el pronóstico de la concentración promedio horaria de PM2.5 para\n"
      "las próximas 96 horas en las estaciones de PM2.5 del Valle de Aburrá.\n\n"
      "* Los modelos se desarrollaron usando información de aerosoles proveniente de CAMS (ECMWF) y\n"
      "meteorológica proveniente de GFS (NCEP). Cada línea punteada indica el pronóstico promedio\n"
      "(a partir de distintos métodos) para cada estación.\n\n")
text2 = plt.figtext(0.94, -0.25, t2,fontsize=11, wrap=True)

path_fig = '/home/jsperezc/jupyter/AQ_Forecast/operational_results/figures/stations/SummaryForecasts_current.png'
plt.savefig(path_fig,bbox_inches='tight')
plt.close('all')


# In[13]:


## FIGURA 4 RESUMEN 24H

cmaps = {'Norte':matplotlib.cm.get_cmap('Purples'),
        'Centro':matplotlib.cm.get_cmap('Greens'),
        'Sur':matplotlib.cm.get_cmap('Blues'),
        'Tráfico':matplotlib.cm.get_cmap('Oranges')}

colores_regiones = {'Norte':'purple',
        'Centro':'green',
        'Sur':'blue',
        'Tráfico':'orange'}


regiones_estaciones = {'Norte':np.array(['BAR-TORR','COP-CVID','BEL-FEVE','MED-VILL']),        'Centro':np.array(['MED-ARAN','MED-SCRI','MED-SELE','MED-BEME','MED-ALTA','MED-TESO']),        'Sur':np.array(['MED-LAYE','ITA-CJUS','ENV-HOSP','ITA-CONC','EST-HOSP','CAL-JOAR','CAL-LASA','SAB-RAME']),        'Tráfico':np.array(['CEN-TRAF','SUR-TRAF'])}

limites_ica = {'C.A. Buena':(0,12.5),
              'C.A. Moderada':(12.5,37.5),
              'C.A. Dañina a grupos sensibles':(37.5,55.5),
              'C.A. Dañina a la salud':(55.5,150.5),
              'C.A. Muy dañina a la salud':(150.5,250.5),
              'C.A. Peligrosa':(250.5,500)}

colores_ica = ['green','yellow','orange','red','purple','brown']

fig = plt.figure(figsize=(11,5))

tables = {}
for region in regiones_estaciones.keys():
    tables[region] = pd.DataFrame()

for i_station,station in enumerate(list(dic_forecasts_stations.keys())):
    if station in regiones_estaciones['Sur']:
        region = 'Sur'
    elif station in regiones_estaciones['Norte']:
        region = 'Norte'
    elif station in regiones_estaciones['Centro']:
        region = 'Centro'
    elif station in regiones_estaciones['Tráfico']:
        region = 'Tráfico'
        
    df_24h = pd.concat([dic_pm_stations[station],dic_forecasts_stations[station]['MEAN']]).mean(axis=1).        rolling(24).mean()[dic_forecasts_stations[station]['MEAN'].index]
    pm2p5_24h = dic_pm_stations[station].rolling(24).mean()
    
    cmap = cmaps[region]
    where_station = np.where(regiones_estaciones[region]==station)[0][0]
    plt.plot(df_24h,        color=cmap((((where_station+1)/(len(regiones_estaciones[region])*2))/0.8)+0.1),        ls = '--',alpha = 0.7)
    plt.plot(pm2p5_24h,        color=cmap((((where_station+1)/(len(regiones_estaciones[region])*2))/0.8)+0.1),        alpha = 0.7,label = station)
    
    tables[region][station] = df_24h
    
plt.axvline(forecast_initial_date,color='k',ls = '--',label='Fecha inicial de pronóstico')
ylim = plt.gca().get_ylim()

#### Background ICA

for i in range(len(colores_ica)):
    plt.fill_between([dates_forecast[0]-dt.timedelta(hours=48),dates_forecast[-1]],
                limites_ica[list(limites_ica.keys())[i]][0],
                limites_ica[list(limites_ica.keys())[i]][1],
                alpha=0.1,color=colores_ica[i],label=list(limites_ica.keys())[i])
    
plt.legend(ncol=3,bbox_to_anchor=(0.98, -0.12),fontsize=13)

plt.xticks(fontsize=(12))
plt.yticks(fontsize=(12))
plt.ylabel('Concentración de PM2.5\n[$\mu g/m^3$]',fontsize=14)
plt.xlim(dates_forecast[0]-dt.timedelta(hours=12),dates_forecast[-1])
plt.grid(ls='--',alpha=0.3)
plt.title('Fecha inicial del pronóstico: '+str(forecast_initial_date)+ ' HL',fontsize=15,loc='left',y = 1.02)

table = pd.DataFrame()
for region in regiones_estaciones.keys():
    table[region] = tables[region].mean(axis=1)
    
    plt.plot(table[region],        color=colores_regiones[region],        alpha = 0.7,label = region)

plt.ylim(ylim[0], ylim[1])


plt.legend(ncol=3,bbox_to_anchor=(0.98, -0.12),fontsize=13)
table = table.iloc[:18]
table = table.round(2)
table['date'] = table.index.to_pydatetime().astype(str)
table['date'] = [table['date'].iloc[i][:-3] for i in range(len(table))]

table = table[['date','Norte','Centro','Sur']]
cell_text = []
for row in range(len(table)):
    cell_text.append(table.iloc[row])

table.columns = ['Fecha','Norte [$\mu g / m^3$]','Centro [$\mu g / m^3$]','Sur [$\mu g / m^3$]']
ptable = plt.table(cellText=cell_text, colLabels=table.columns, bbox=(1,0,1,1),edges='vertical',
                  colLoc = 'center',rowLoc='center')
ptable.auto_set_font_size(False)
ptable.set_fontsize(12)
ptable.auto_set_column_width((-1, 0, 1, 2, 3))

for (row, col), cell in ptable.get_celld().items():
    if (row == 0):
        cell.set_text_props(fontproperties=FontProperties(weight='bold',size=12))

text = plt.figtext(0.94, 0.91, 'Pronóstico de las próximas 18 horas en promedio en cada sub-región.',fontsize=15)
t2 = ("* Este producto indica el pronóstico de la concentración de PM2.5 en promedios de 24 horas para\n"
      "las próximas 96 horas en las estaciones de PM2.5 del Valle de Aburrá.\n\n"
      "* Los modelos se desarrollaron usando información de aerosoles proveniente de CAMS (ECMWF) y\n"
      "meteorológica proveniente de GFS (NCEP). Cada línea punteada indica el pronóstico promedio\n"
      "(a partir de distintos métodos) para cada estación.\n\n")
text2 = plt.figtext(0.94, -0.25, t2,fontsize=11, wrap=True)

path_fig = '/home/jsperezc/jupyter/AQ_Forecast/operational_results/figures/stations/SummaryForecasts_24h_current.png'
plt.savefig(path_fig,bbox_inches='tight')
plt.close('all')

sal_path = '/var/www/jhayron/AQ_Forecast/operational_results/figures/stations/'

os.system('scp /home/jsperezc/jupyter/AQ_Forecast/operational_results/figures/stations/*.png jsperezc@siata.gov.co:'+\
    sal_path)



