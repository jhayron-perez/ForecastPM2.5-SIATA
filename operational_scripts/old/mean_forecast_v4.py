import sys
sys.path.append('/home/calidadaire/Paquete/')

import airquality.read_data as read

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from matplotlib.font_manager import FontProperties

datetime_now = dt.datetime.now()
forecast_initial_date = datetime_now.replace(minute = 0,second=0,microsecond=0)# - dt.timedelta(hours=1)
gfs_path = '/var/data1/AQ_Forecast_DATA/operational/GFS/'
cams_path = '/var/data1/AQ_Forecast_DATA/operational/CAMS/'

pm2p5 = forecasting.get_pm2p5_for_forecast(datetime_now)
#print(pm2p5)
df_CAMS,df_optimal_cams = forecasting.get_cams_for_forecast(forecast_initial_date,cams_path,operational = True)
df_GFS,df_optimal_gfs = forecasting.get_gfs_for_forecast(forecast_initial_date,gfs_path,operational = True)

#print(forecast_initial_date)
forecasts = pd.DataFrame(index = df_CAMS['pm2p5_cams'][forecast_initial_date:forecast_initial_date+\
                                                  dt.timedelta(hours = 95)].index)

model_names = ['LR','MLP','RF','SVR','GB','KNN','CAMS']
#model_names = ['LR','MLP','RF','SVR','KNN','CAMS']

for model_name in model_names:
    if model_name=='CAMS':
        vals_cams = df_CAMS['pm2p5_cams'][forecast_initial_date:forecast_initial_date+\
                                                 dt.timedelta(hours = 95)].values
#         for i in range(0,96):
#             folder_mos = '/var/data1/AQ_Forecast_DATA/trained_models/mos_operators/'
#             with open(folder_mos+model_name+'lt_'+str(i+1).zfill(2)+'_v2.mos','rb') as f:
#                 mos_temp = pickle.load(f)
#             vals_cams[i] = mos_temp(vals_cams[i])
        forecasts[model_name] = vals_cams
    else:
        forecasts[model_name] = forecasting.forecast_pm2p5_v4(forecast_initial_date,pm2p5,df_optimal_cams,\
            df_optimal_gfs,model_name=model_name,mos_correction=False)
        
forecasts['MEAN'] = np.mean(forecasts.iloc[:,:],axis = 1)
forecasts['MIN'] = np.min(forecasts.iloc[:,:-1],axis = 1)
forecasts['MAX'] = np.max(forecasts.iloc[:,:-2],axis = 1)

############# PLOT 1 ################

fig = plt.figure(figsize=(11,5))

dates_forecast = forecasts.index
for model_name in forecasts.keys()[:-4]:
    if model_name == 'NN':
        plt.plot(dates_forecast,forecasts[model_name].values,label = 'MLP',ls='--',alpha=0.4)
    else:
        plt.plot(dates_forecast,forecasts[model_name].values,label = model_name,ls='--',alpha=0.4)
    
plt.plot(dates_forecast,forecasts['CAMS'].values,label = 'CAMS Corregido (ECMWF)',ls='-.',alpha=0.5,color='gray')
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
plt.title('Fecha inicial del pronóstico: '+str(forecast_initial_date)+ ' HL',fontsize=15,loc='left',y = 1.02)

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
      "estadístico distinto (LR, KNN, GB, SVR, MLP, RF), y su dispersión es proporcional a la\n"
      "incertidumbre asociada.\n\n")
text2 = plt.figtext(0.94, -0.25, t2,fontsize=11, wrap=True)

path_fig = '/home/jsperezc/jupyter/AQ_Forecast/operational_results/figures/mean/archive/Fc_Mean_'+\
    str(forecast_initial_date).replace(':','').replace(' ','_').replace('-','')+'.png'
# plt.show()
plt.savefig(path_fig,bbox_inches='tight')
os.system('cp '+path_fig+' '+path_fig.replace(str(forecast_initial_date).replace(':','').\
    replace(' ','_').replace('-',''),'Current').replace('/archive',''))
path_1 = path_fig
path_2 = path_fig.replace(str(forecast_initial_date).replace(':','').\
    replace(' ','_').replace('-',''),'Current').replace('/archive','')


############# PLOT 2 ################

fig = plt.figure(figsize=(12,12))


ax1 = fig.add_subplot(211)
dates_forecast = forecasts.index

# cams = cams.loc[dates_forecast[-1]dates_forecast[-1]]

plt.plot(df_CAMS.aod,label = 'AOD Total',alpha=1,color='darkblue')
plt.plot(df_CAMS.omaod,label = 'AOD Materia Orgánica',ls='--',alpha=1,color='green')
plt.plot(df_CAMS.suaod,label = u'AOD Sulfato',ls='--',alpha=1,color='gray')
# plt.plot(cams.bcaod550,label = 'AOD Black Carbon',ls='--',alpha=1,color='black')
# plt.plot(cams.duaod550,label = 'AOD Dust',ls='--',alpha=1,color='darkorange')
# plt.plot(cams.ssaod550,label = 'AOD Sea Salt',ls='--',alpha=1,color='darkgray')

plt.axvline(forecast_initial_date,color='k',ls = '--',label='Fecha actual')
# plt.plot(pm2p5[forecast_initial_date-dt.timedelta(hours=12):],lw=1.4,color='k',label='Observations')

plt.legend(ncol=1,fontsize=13)

plt.xticks(fontsize=(12))
plt.yticks(fontsize=(12))
plt.ylabel('AOD (550nm)',fontsize=14)

plt.xlim(dates_forecast[0]-dt.timedelta(hours=12),dates_forecast[-1])
plt.grid(ls='--',alpha=0.3)

plt.title('Fecha inicial del pronóstico de CAMS (ECMWF): '+str(df_CAMS.index[0])+ ' HL',fontsize=14,loc='left')

ax2 = fig.add_subplot(212)
plt.plot(df_CAMS.bcaod,label = u'AOD Carbón Negro',ls='--',alpha=1,color='black')
plt.plot(df_CAMS.duaod,label = u'AOD Polvo',ls='--',alpha=1,color='darkorange')
plt.plot(df_CAMS.ssaod,label = u'AOD Sal Marina',ls='--',alpha=1,color='darkgray')

plt.plot(df_CAMS.niaod,label = u'AOD Nitrato',ls='--',alpha=1,color='firebrick')
plt.plot(df_CAMS.amaod,label = u'AOD Amonia',ls='--',alpha=1,color='rebeccapurple')

plt.axvline(forecast_initial_date,color='k',ls = '--',label='Fecha actual')
# plt.plot(pm2p5[forecast_initial_date-dt.timedelta(hours=12):],lw=1.4,color='k',label='Observations')

plt.legend(fontsize=13)

plt.xticks(fontsize=(12))
plt.yticks(fontsize=(12))
plt.ylabel('AOD (550nm)',fontsize=14)

plt.xlim(dates_forecast[0]-dt.timedelta(hours=12),dates_forecast[-1])
plt.grid(ls='--',alpha=0.3)
path_fig = '/home/jsperezc/jupyter/AQ_Forecast/operational_results/figures/mean/archive/Fc_AOD_'+\
    str(forecast_initial_date).replace(':','').replace(' ','_').replace('-','')+'.png'
# plt.show()
plt.savefig(path_fig,bbox_inches='tight')
os.system('cp '+path_fig+' '+path_fig.replace(str(forecast_initial_date).replace(':','').\
                                              replace(' ','_').replace('-',''),'Current').replace('/archive',''))

path_3 = path_fig
path_4 = path_fig.replace(str(forecast_initial_date).replace(':','').\
                                              replace(' ','_').replace('-',''),'Current').replace('/archive','')


### Guardar tabla:
path_table = '/home/jsperezc/jupyter/AQ_Forecast/operational_results/tables/mean/archive/Fc_AOD_'+\
    str(forecast_initial_date).replace(':','').replace(' ','_').replace('-','')+'.csv'
forecasts.to_csv(path_table)
forecasts.to_csv(path_table.replace(str(forecast_initial_date).replace(':','').\
                                              replace(' ','_').replace('-',''),'Current').replace('/archive',''))

path_5 = path_table
path_6 = path_table.replace(str(forecast_initial_date).replace(':','').\
                                              replace(' ','_').replace('-',''),'Current').replace('/archive','')


### FIGURA EVALUACIÓN OPERACIONAL

files_forecasts = np.sort(glob.glob\
    ('/home/jsperezc/jupyter/AQ_Forecast/operational_results/tables/mean/archive/*.csv'))

dates_past_forecasts = np.array([dt.datetime.strptime(files_forecasts[i].split('/')[-1],'Fc_AOD_%Y%m%d_%H%M%S.csv') for i in \
    range(len(files_forecasts))])

files_forecasts = files_forecasts[dates_past_forecasts >= \
    forecast_initial_date-dt.timedelta(hours=96)]
dates_past_forecasts = dates_past_forecasts[dates_past_forecasts >= \
    forecast_initial_date-dt.timedelta(hours=96)]

dic_past_forecast = {}

for i in range(len(files_forecasts)):
    dic_past_forecast[str(dates_past_forecasts[i])] = \
        pd.read_csv(files_forecasts[i],index_col=0,parse_dates=True)[['MEAN','MIN','MAX']]
    
fig = plt.figure(figsize=(11,5))

for i in range(len(files_forecasts)):
    plt.plot(dic_past_forecast[str(dates_past_forecasts[i])]['MEAN'],alpha=0.1,color='teal')
    plt.fill_between(dic_past_forecast[str(dates_past_forecasts[i])].index,
        dic_past_forecast[str(dates_past_forecasts[i])]['MIN'],
        dic_past_forecast[str(dates_past_forecasts[i])]['MAX'],
        alpha=0.04,color='skyblue')
    
plt.plot(dates_forecast,np.mean(forecasts.iloc[:,:-4],axis = 1),label = 'Media del ensamble',
         color='teal',lw=1.4,alpha=1)
plt.fill_between(dates_forecast,np.min(forecasts.iloc[:,:-4],axis = 1),np.max(forecasts.iloc[:,:-2],axis = 1),
                alpha=0.04,color='skyblue',label='Rango del ensamble')
plt.axvline(forecast_initial_date,color='k',ls = '--',label='Fecha del último pronóstico')
    
plt.plot(pm2p5,color='k',label='Observaciones')
plt.xlim(dic_past_forecast[str(dates_past_forecasts[0])].index[0],
         dic_past_forecast[str(dates_past_forecasts[-1])].index[-1])

plt.legend(ncol=2,bbox_to_anchor=(0.84, -0.12),fontsize=13)
plt.xticks(fontsize=(12))
plt.yticks(fontsize=(12))
plt.ylabel('Concentración de PM2.5\n[$\mu g/m^3$]',fontsize=14)
# plt.xlim(dates_forecast[0]-dt.timedelta(hours=12),dates_forecast[-1])
plt.grid(ls='--',alpha=0.3)
plt.title('Pronósticos inicializados en las últimas 96 horas',fontsize=15,loc='left',y = 1.02)
plt.grid(ls='--',alpha=0.3)
# plt.show()
path_fig = '/home/jsperezc/jupyter/AQ_Forecast/operational_results/figures/mean/archive/Fc_Eval_'+\
    str(forecast_initial_date).replace(':','').replace(' ','_').replace('-','')+'.png'
plt.savefig(path_fig,bbox_inches='tight')
os.system('cp '+path_fig+' '+path_fig.replace(str(forecast_initial_date).replace(':','').\
                                              replace(' ','_').replace('-',''),'Current').replace('/archive',''))

path_7 = path_fig
path_8 = path_fig.replace(str(forecast_initial_date).replace(':','').\
                                              replace(' ','_').replace('-',''),'Current').replace('/archive','')


### FIGURA ICA:

forecast_24h_mean = copy.deepcopy(forecasts)
for model_name in forecasts.keys():
    forecast_24h_mean[model_name] = pd.concat([pm2p5['PM2.5'],forecasts[model_name]],axis=0).\
        rolling(24).mean()[forecasts.index]
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
plt.title('Fecha inicial de pronóstico '+str(forecast_initial_date)+ ' HL',fontsize=15,loc='left',y = 1.02)

#### Background ICA

for i in range(len(colores_ica)):
    plt.fill_between([dates_forecast[0]-dt.timedelta(hours=48),dates_forecast[-1]],
                limites_ica[list(limites_ica.keys())[i]][0],
                limites_ica[list(limites_ica.keys())[i]][1],
                alpha=0.1,color=colores_ica[i],label=list(limites_ica.keys())[i])
    
plt.ylim(np.max([pd.concat([pm2p5_24h_mean,forecast_24h_mean]).min().min()-10,0]),\
        np.max([pd.concat([pm2p5_24h_mean,forecast_24h_mean]).max().max()+10,0]))

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
      "estadístico distinto (LR, KNN, GB, SVR, MLP, RF), y su dispersión es proporcional a la\n"
      "incertidumbre asociada.\n\n")
text2 = plt.figtext(0.94, -0.25, t2,fontsize=11, wrap=True)
path_fig = '/home/jsperezc/jupyter/AQ_Forecast/operational_results/figures/mean/archive/Fc_Mean24h_'+\
    str(forecast_initial_date).replace(':','').replace(' ','_').replace('-','')+'.png'
# plt.show()
plt.savefig(path_fig,bbox_inches='tight')
os.system('cp '+path_fig+' '+path_fig.replace(str(forecast_initial_date).replace(':','').\
                                              replace(' ','_').replace('-',''),'Current').replace('/archive',''))

path_9 = path_fig
path_10 = path_fig.replace(str(forecast_initial_date).replace(':','').\
                                              replace(' ','_').replace('-',''),'Current').replace('/archive','')


### Copia a SAL:

local_path = '/home/jsperezc/jupyter/AQ_Forecast/operational_results/'
sal_path = '/var/www/jhayron/AQ_Forecast/operational_results/'


os.system('scp '+path_1+' jsperezc@siata.gov.co:'+path_1.replace(local_path,sal_path))
os.system('scp '+path_2+' jsperezc@siata.gov.co:'+path_2.replace(local_path,sal_path))
os.system('scp '+path_3+' jsperezc@siata.gov.co:'+path_3.replace(local_path,sal_path))
os.system('scp '+path_4+' jsperezc@siata.gov.co:'+path_4.replace(local_path,sal_path))
os.system('scp '+path_5+' jsperezc@siata.gov.co:'+path_5.replace(local_path,sal_path))
os.system('scp '+path_6+' jsperezc@siata.gov.co:'+path_6.replace(local_path,sal_path))
os.system('scp '+path_7+' jsperezc@siata.gov.co:'+path_7.replace(local_path,sal_path))
os.system('scp '+path_8+' jsperezc@siata.gov.co:'+path_8.replace(local_path,sal_path))
os.system('scp '+path_9+' jsperezc@siata.gov.co:'+path_9.replace(local_path,sal_path))
os.system('scp '+path_10+' jsperezc@siata.gov.co:'+path_10.replace(local_path,sal_path))
