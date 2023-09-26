def plot_ForeICA24h_pob(date_forecast, path_output):
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
    
    #Fecha para la lectura del pronóstico
    year = str(date_forecast.year)
    month = str(date_forecast.month).zfill(2)
    day = str(date_forecast.day).zfill(2)
    hour = str(date_forecast.hour).zfill(2)
    
    #Definiendo estaciones
    AQStats = ["ITA-CJUS", "ITA-CONC", "MED-LAYE", "CAL-JOAR", "EST-HOSP", "MED-ALTA",
               "MED-VILL", "BAR-TORR", "COP-CVID", "MED-BEME", "MED-TESO", "MED-SCRI",
               "MED-ARAN", "BEL-FEVE", "ENV-HOSP", "SAB-RAME", "MED-SELE"]
    
    #Para leer información del pronóstico
    datei_for = '{0}-{1}-{2} {3}:00'.format(year, month, day, hour)
    #Para leer información de las observaciones (hasta antes de la hora de inicialización)
    datef_obs = (pd.to_datetime(datei_for) - pd.Timedelta(hours=1)).strftime('%Y-%m-%d %H:%M')
    datei_obs = (pd.to_datetime(datef_obs) - pd.Timedelta(days=1)).strftime('%Y-%m-%d %H:%M')

    #Cargando datos requeridos calidad del aire
    DataObs_PM25 = aqread.PM25(Fechai=datei_obs, Fechaf=datef_obs, estaciones=AQStats)
    DataObs_PM25 = DataObs_PM25.data

    #Creando DataFrame para almacenar la información
    DFTotal_PM25 = pd.DataFrame(DataObs_PM25.mean(axis=0))
    #Dando estructura al DF
    for idx in range(1, 25):
        DFTotal_PM25[idx] = np.nan
    
    
    #Leyendo archivo del pronóstico (con base en la fecha)
    filename = '{0}{1}{2}_{3}.csv'.format(year, month, day, hour)

    #Recorriendo las estaciones
    for aqsta in AQStats:
        #Directorio de entrada
        dirin_data = '/var/data1/AQ_Forecast_DATA/results/{0}/csv/'.format(aqsta)
        #Obteniendo información
        try:
            DataFore_PM25 = pd.read_csv(dirin_data + filename, index_col=[0], parse_dates=[0],
                                        infer_datetime_format=True)
            DataFore_PM25 = DataFore_PM25.mean(axis=1).iloc[:24]
        except:
            continue

        #Guardando información en DataFrame
        DFTotal_PM25.loc[aqsta, range(1, 25)] = DataFore_PM25.values

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
    text1 = plt.figtext(0.125, 0.15, t1, fontsize=11, wrap=True, horizontalalignment='left')


    #Guardando y mostrando la figura
    plt.savefig(dirout_figs, dpi=300, bbox_inches='tight')
    plt.close()   
    

#### ------------------------------------- #####



# %%
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
from matplotlib.font_manager import FontProperties

datetime_now = dt.datetime.now()
forecast_initial_date = datetime_now.replace(minute = 0,second=0,microsecond=0)# - dt.timedelta(hours=1)
gfs_path = '/var/data1/AQ_Forecast_DATA/operational/GFS/'
cams_path = '/var/data1/AQ_Forecast_DATA/operational/CAMS/'

pm2p5 = preprocessing.get_pm2p5_for_forecast(datetime_now)
#print(pm2p5)
df_CAMS,df_optimal_cams = preprocessing.get_cams_for_forecast(forecast_initial_date,cams_path,operational = True)
df_GFS,df_optimal_gfs = preprocessing.get_gfs_for_forecast(forecast_initial_date,gfs_path,operational = True)

#print(forecast_initial_date)
forecasts = pd.DataFrame(index = df_CAMS['pm2p5_cams'][forecast_initial_date:forecast_initial_date+\
                                                  dt.timedelta(hours = 95)].index)

model_names = ['LR','MLP','RF','SVR','GB','KNN','CAMS']
model_names = ['LR','MLP','RF','SVR','KNN','CAMS']
model_names = ['LR','MLP','RF','SVR','CAMS']

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


#Obtener información de cada una de las estaciones (para esto hay que aprender a guardar los csvs)
#

# %%
#Test para graficar figura del ICA
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
#dirout_figs = path_output
date_forecast = pd.to_datetime('2023-02-21 17:00')

#Fecha para la lectura del pronóstico
year = str(date_forecast.year)
month = str(date_forecast.month).zfill(2)
day = str(date_forecast.day).zfill(2)
hour = str(date_forecast.hour).zfill(2)

#Definiendo estaciones
AQStats = ["ITA-CJUS", "ITA-CONC", "MED-LAYE", "CAL-JOAR", "EST-HOSP", "MED-ALTA",
            "MED-VILL", "BAR-TORR", "COP-CVID", "MED-BEME", "MED-TESO", "MED-SCRI",
            "MED-ARAN", "BEL-FEVE", "ENV-HOSP", "SAB-RAME", "MED-SELE"]

#Para leer información del pronóstico
datei_for = '{0}-{1}-{2} {3}:00'.format(year, month, day, hour)
#Para leer información de las observaciones (hasta antes de la hora de inicialización)
datef_obs = (pd.to_datetime(datei_for) - pd.Timedelta(hours=1)).strftime('%Y-%m-%d %H:%M')
datei_obs = (pd.to_datetime(datef_obs) - pd.Timedelta(days=1)).strftime('%Y-%m-%d %H:%M')

#Cargando datos requeridos calidad del aire
DataObs_PM25 = aqread.PM25(Fechai=datei_obs, Fechaf=datef_obs, estaciones=AQStats)
DataObs_PM25 = DataObs_PM25.data


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
        DataFore_PM25_sta = DataFore_PM25.iloc[:24].loc[:, 'MEAN'].copy()
        #Obteniendo observaciones de las últimas 23 horas
        DataObs_PM25_sta = DataObs_PM25.loc[:, aqsta].iloc[2:].copy()
        
        #Uniendo información y obteniendo media móvil de 24 horas
        Concat_PM25ObsFor_sta = pd.concat([DataObs_PM25_sta, DataFore_PM25_sta])
        DataFore_PM25_sta_24h = Concat_PM25ObsFor_sta.rolling(window=24, min_periods=18).mean().iloc[-24:]

    except:
        continue
    
    #Guardando información en DataFrame
    DFTotal_PM25.loc[aqsta, range(1, 25)] = DataFore_PM25_sta_24h.values
    
    

# %%

