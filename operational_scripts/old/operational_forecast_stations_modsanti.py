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
import preprocessing
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


#Estableciendo límite de procesadores para scipy y scikit-learn (error KNN)
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1, user_api='blas')
threadpool_limits(limits=1, user_api='openmp')


#Parámetros para el pronóstico
stations = ["ITA-CJUS", "CAL-LASA", "ITA-CONC", "MED-LAYE", "CAL-JOAR", "EST-HOSP", "MED-ALTA", "MED-VILL", 
            "BAR-TORR", "COP-CVID", "MED-BEME", "MED-TESO", "MED-SCRI", "MED-ARAN", "BEL-FEVE", "ENV-HOSP", 
            "SAB-RAME", "MED-SELE","CEN-TRAF"]

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


        pm2p5 = preprocessing.get_pm2p5_for_forecast(datetime_now,station_name = station)

        df_CAMS,df_optimal_cams = forecasting.get_cams_for_forecast(forecast_initial_date,cams_path,operational = True,
                                                                   latlon = (lat_est,lon_est))
        df_GFS,df_optimal_gfs = forecasting.get_gfs_for_forecast(forecast_initial_date,gfs_path,operational = True,
                                                                latlon = (lat_est,lon_est))

        forecasts = pd.DataFrame(index = df_CAMS['pm2p5_cams'][forecast_initial_date:forecast_initial_date+                                                          dt.timedelta(hours = 95)].index)

        model_names = ['LR','MLP','RF','SVR','KNN','CAMS']
        #model_names = ['LR','MLP','RF','SVR','CAMS']

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
        try:os.mkdir('/var/data1/AQ_Forecast_DATA/results_oldv/tables/stations/archive/'+station+'/')
        except: pass
        path_table = '/var/data1/AQ_Forecast_DATA/results_oldv/tables/stations/archive/'+            station+'/'+'Fc_'+station+'_'+            str(forecast_initial_date).replace(':','').replace(' ','_').replace('-','')+'.csv'
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

    path_fig = '/var/data1/AQ_Forecast_DATA/results_oldv/figures/stations/Fc_'+name_station+'_'+        'current.png'
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
    path_fig = '/var/data1/AQ_Forecast_DATA/results_oldv/figures/stations/Fc24h_'+name_station+'_'+        'current.png'
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

path_fig = '/var/data1/AQ_Forecast_DATA/results_oldv/figures/stations/SummaryForecasts_current.png'
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

path_fig = '/var/data1/AQ_Forecast_DATA/results_oldv/figures/stations/SummaryForecasts_24h_current.png'
plt.savefig(path_fig,bbox_inches='tight')
plt.close('all')

# %%
### FUNCIÓN PARA GRAFICAR FIGURA DE CUADRITOS (PRONÓSTICO)
def plot_ForeICA24h_4old(date_forecast, path_output, AQStats, ytext=0.15):
    #Importando librerías
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import datetime as dt

    #Importando airquality
    import sys
    sys.path.append('/home/calidadaire/Paquete/')
    import airquality.read_data as aqread

    #Definiendo directorio de salida
    dirout_figs = path_output
    
    ### ---- Solo para pruebas ----
    #date_forecast = pd.to_datetime('2023-02-21 17:00')
    #AQStats = ["ITA-CJUS", "ITA-CONC", "MED-LAYE", "CAL-JOAR", "EST-HOSP", "MED-ALTA",
    # "MED-VILL", "BAR-TORR", "COP-CVID", "MED-BEME", "MED-TESO", "MED-SCRI",
    # "MED-ARAN", "BEL-FEVE", "ENV-HOSP", "SAB-RAME", "MED-SELE","CEN-TRAF"]
    #dirout_figs = './TestICA.png'
    #ytext = 0.12
    

    #Fecha para la lectura del pronóstico
    year = str(date_forecast.year)
    month = str(date_forecast.month).zfill(2)
    day = str(date_forecast.day).zfill(2)
    hour = str(date_forecast.hour).zfill(2)

    #Para leer información del pronóstico
    datei_for = '{0}-{1}-{2} {3}:00'.format(year, month, day, hour)
    #Para leer información de las observaciones (hasta antes de la hora de inicialización)
    datef_obs = (pd.to_datetime(datei_for) - pd.Timedelta(hours=1)).strftime('%Y-%m-%d %H:%M')
    datei_obs = (pd.to_datetime(datef_obs) - pd.Timedelta(days=1)).strftime('%Y-%m-%d %H:%M')

    #Cargando datos requeridos calidad del aire
    DataObs_PM25 = aqread.PM25(Fechai=datei_obs, Fechaf=datef_obs, estaciones=AQStats)
    DataObs_PM25 = DataObs_PM25.data
    
    #Definiendo modelos para promediar (ensamble)
    model_names_ensamble = model_names.copy()
    model_names_ensamble.remove('CAMS')


    #Creando DataFrame para almacenar la información
    DFTotal_PM25 = pd.DataFrame(DataObs_PM25.mean(axis=0))
    #Dando estructura al DF
    for idx in range(1, 25):
        DFTotal_PM25[idx] = np.nan


    #Recorriendo las estaciones
    for aqsta in AQStats:
        #Directorio de entrada
        dirin_data = '/var/data1/AQ_Forecast_DATA/results_oldv/tables/stations/archive/{0}/'.format(aqsta)
        #Leyendo archivo del pronóstico (con base en la fecha y estación)
        filename = 'Fc_{0}_{1}{2}{3}_{4}0000.csv'.format(aqsta, year, month, day, hour)
        
        #dirin_data = '/var/data1/AQ_Forecast_DATA/results_oldv/{0}/csv/'.format(aqsta)
        #Obteniendo información
        try:
            DataFore_PM25 = pd.read_csv(dirin_data + filename, index_col=[0], parse_dates=[0],
                                        infer_datetime_format=True)
            #Limitando a 24 horas y obteniendo el promedio de la estación
            
            DataFore_PM25_sta = DataFore_PM25.iloc[:23].loc[:, model_names_ensamble].mean(axis=1).copy()
            #Obteniendo observaciones de las últimas 23 horas
            DataObs_PM25_sta = DataObs_PM25.loc[:, aqsta].iloc[1:].copy()
            
            #Uniendo información y obteniendo media móvil de 24 horas
            Concat_PM25ObsFor_sta = pd.concat([DataObs_PM25_sta, DataFore_PM25_sta])
            DataFore_PM25_sta_24h = Concat_PM25ObsFor_sta.rolling(window=24, min_periods=18).mean().iloc[-24:]

        except:
            continue
        
        #Guardando información en DataFrame
        DFTotal_PM25.loc[aqsta, range(1, 25)] = DataFore_PM25_sta_24h.values

    #Ordenando dataframe en función de la concentración pronosticada a 6 horas
    DFTotal_PM25.sort_values(6, ascending=False, inplace=True)
    
    
    #GRAFICANDO INFORMACIÓN
    #Definiendo paleta de colores en función del ICA
    import matplotlib as mpl
    cmap = mpl.colors.ListedColormap(['#00ab5c','#ffff01','#fea500','#fe0000','#873ac0'])
    ICAval = [0, 12.6, 37.6, 55.6, 150, 501]
    norm = mpl.colors.BoundaryNorm(ICAval, cmap.N)

    #Creando figura
    fig, ax = plt.subplots(figsize=(12, 12))

    #Graficando información
    ax.imshow(DFTotal_PM25.values, cmap=cmap, norm=norm)

    #Parámetros adicionales
    ax.set_yticks(range(len(DFTotal_PM25)))
    ax.set_yticklabels(DFTotal_PM25.index)
    ax.set_xticks(range(0, 25))
    ax.set_xticklabels(['Now'] + ['+'+str(i) for i in range(1, 25)])
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_aspect(0.7)
    ax.axvline(0.5, color='black', linestyle=':')
    ax.set_xlabel('Horas (pronóstico)', fontsize=12, labelpad=10)
    ax.xaxis.set_label_position('top')


    #Añadiendo grillas
    # Minor ticks
    ax.set_xticks(np.arange(-.5, 25, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(DFTotal_PM25), 1), minor=True)
    #Grid
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    #Eliminando minor ticks
    for tick in ax.xaxis.get_minor_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    for tick in ax.yaxis.get_minor_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    #Estableciendo fecha inicial del pronóstico
    plt.title('Fecha inicial del pronóstico: '+str(date_forecast)+ ' HL',
              fontsize=14, loc='left')
    
    
    #Definiendo listas con la información requerida para la leyenda
    Labels = ['Buena', 'Aceptable', 'Dañina grupos sensibles', 'Dañina', 'Muy dañina']
    Colors = ['#00ab5c','#ffff01','#fea500','#fe0000','#873ac0']

    handles = [Patch(facecolor=color, label=label) for label, color in zip(Labels, Colors)]

    #Añadiendo leyenda
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0., -0.01), ncol=5, fontsize=12.5,
              edgecolor='white', handletextpad=0.6)
    
    #Añadiendo texto
    t1 = ("* Esta figura muestra el pronóstico del Índice de Calidad del Aire (ICA) para diferentes estaciones poblacionales\nde calidad del aire del Valle de Aburrá.\n\n"
          "* Now se refiere al ICA de las últimas 24 horas (observado).\n\n"
          "* Los pronósticos se indican con el signo + y la hora en el futuro a la que se está pronosticando.\nPor ejemplo: +6 indica el pronóstico a 6 horas.\n\n")
    text1 = plt.figtext(0.125, ytext, t1, fontsize=11, wrap=True, horizontalalignment='left')


    #Guardando y mostrando la figura
    plt.savefig(dirout_figs, dpi=300, bbox_inches='tight')
    plt.close()   

# %%

### FIGURA DE PRONÓSTICO DE CALIDAD DEL AIRE (CUADRITOS) PARA TODAS LAS ESTACIONES
#Definiendo estaciones
AQStats = ["ITA-CJUS", "ITA-CONC", "MED-LAYE", "CAL-JOAR", "EST-HOSP", "MED-ALTA",
            "MED-VILL", "BAR-TORR", "COP-CVID", "MED-BEME", "MED-TESO", "MED-SCRI",
            "MED-ARAN", "BEL-FEVE", "ENV-HOSP", "SAB-RAME", "MED-SELE","CEN-TRAF"]
#Definiendo path
path_fig = '/var/data1/AQ_Forecast_DATA/results_oldv/figures/stations/ForecastICA24h_SummaryAll_current.png'
#Graficando
plot_ForeICA24h_4old(forecast_initial_date, path_fig, AQStats, ytext=0.12)
#Copiando figura al archivo
path_copyfig = '/var/data1/AQ_Forecast_DATA/results_oldv/figures/mean/archive/ForecastICA24h_SummaryAll_'+\
    str(forecast_initial_date).replace(':','').replace(' ','_').replace('-','')+'.png'
os.system('cp '+path_fig+' '+path_copyfig)



### FIGURA DE PRONÓSTICO DE CALIDAD DEL AIRE (CUADRITOS) PARA LAS ESTACIONES POECA
#Definiendo estaciones
AQStats = ["ITA-CJUS", "ITA-CONC", "MED-LAYE", "CAL-JOAR", "EST-HOSP", "MED-ALTA",
            "MED-VILL", "BAR-TORR", "COP-CVID", "MED-BEME", "MED-TESO", "MED-SCRI",
            "MED-ARAN", "BEL-FEVE", "ENV-HOSP", "SAB-RAME", "MED-SELE"]

#Definiendo path
path_fig = '/var/data1/AQ_Forecast_DATA/results_oldv/figures/stations/ForecastICA24h_SummaryPOECA_current.png'
#Graficando
plot_ForeICA24h_4old(forecast_initial_date, path_fig, AQStats)
#Copiando figura al archivo
path_copyfig = '/var/data1/AQ_Forecast_DATA/results_oldv/figures/mean/archive/ForecastICA24h_SummaryPOECA_'+\
    str(forecast_initial_date).replace(':','').replace(' ','_').replace('-','')+'.png'
os.system('cp '+path_fig+' '+path_copyfig)



#sal_path = '/var/www/jhayron/AQ_Forecast/operational_results/figures/stations/'

#os.system('scp /home/jsperezc/jupyter/AQ_Forecast/operational_results/figures/stations/*.png jsperezc@siata.gov.co:'+\
#    sal_path)


#Copiando información a SAL
os.system('scp -r /var/data1/AQ_Forecast_DATA/results_oldv/figures/stations/*.png calidadaire@siata.gov.co:/var/www/CalidadAire/Pronostico_PM25_OldV/figures/stations')



print("DONE!!")
