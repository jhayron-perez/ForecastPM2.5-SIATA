import os

os.environ['OPENBLAS_NUM_THREADS'] = '20'
os.environ['MKL_NUM_THREADS'] = '20'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from scipy import interpolate
import random
import traceback

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import sys
sys.path.insert(1,'/home/jsperezc/jupyter/AQ_Forecast/functions/')
from preprocessing import *
import forecasting
from plotting import *

stations = ["ITA-CJUS", "ITA-CONC", "MED-LAYE", "CAL-JOAR", "EST-HOSP", "MED-ALTA", "MED-VILL",
            "BAR-TORR", "COP-CVID", "MED-BEME", "MED-TESO", "MED-SCRI", "MED-ARAN", "BEL-FEVE", "ENV-HOSP", 
            "SAB-RAME", "MED-SELE","CEN-TRAF","SUR-TRAF"]
stations = ["ITA-CJUS", "ITA-CONC", "MED-LAYE", "CAL-JOAR", "EST-HOSP", "MED-ALTA", "MED-VILL",
            "BAR-TORR", "COP-CVID", "MED-BEME", "MED-TESO", "MED-SCRI", "MED-ARAN", "BEL-FEVE", "ENV-HOSP", 
            "SAB-RAME", "MED-SELE","CEN-TRAF"]

models = ['GB_MO','GB_CH','RF_MO','RF_CH']#,'LR_MO','LR_RC']

dic_forecasts = {}
dic_probs_dataframes = {}
dic_pm2p5 = {}
for i_station in range(len(stations)):
# for i_station in [0]:
    try:
        datetime_now = dt.datetime.now()#-dt.timedelta(hours=1)
        station = stations[i_station]
        print(station)
        forecast_initial_date = datetime_now.replace(minute = 0,second=0,microsecond=0)

        ### Collect inputs ###
        index_future, x_temp, pm2p5, df_GFS, df_CAMS, df_IFRP = get_inputs_for_forecast(station, forecast_initial_date)
        ### Predict ###
        df_forecasts = forecasting.forecast_station(x_temp,index_future,station,models)
        ### Predict probabilities RF-MO ###
        members_array, members_df = forecasting.predict_probabilities_mo(station,x_temp,index_future)

        #### Define paths ####
        str_initial_date = str(forecast_initial_date).replace('-','').replace(' ','_')[:-6]
        folder_outputs = '/var/data1/AQ_Forecast_DATA/results/'+station+'/'

        if os.path.isdir(folder_outputs)==False:
            os.mkdir(folder_outputs)

        folder_output_texts = folder_outputs+'csv/'
        folder_output_figures = folder_outputs+'png/'

        if os.path.isdir(folder_output_texts)==False:
            os.mkdir(folder_output_texts)

        if os.path.isdir(folder_output_figures)==False:
            os.mkdir(folder_output_figures)

        folder_var_texts = '/var/data1/AQ_Forecast_DATA/results/var/csv/'
        folder_var_figures = '/var/data1/AQ_Forecast_DATA/results/var/png/'

        #### Save forecasts csv ####

        ### Archive ###
        df_forecasts.to_csv(folder_output_texts+str_initial_date+'.csv')
        ### var ###
        df_forecasts.to_csv(folder_var_texts+station+'.csv')

        #### Save percentiles csv ####

        ### Archive ###
        members_df.to_csv(folder_output_texts+str_initial_date+'_prob.csv')
        ### var ###
        members_df.to_csv(folder_var_texts+station+'_prob.csv')

        ### FIGURA 1: Pronóstico individual ###
        path_individual_figure = folder_output_figures+'Fc24h_'+str_initial_date+'.png'
        plot_forecasts_24h_individual_operational(station,df_forecasts,pm2p5,\
            path_individual_figure,probabilities=True,df_probs=members_df)

        path_individual_figure_var = folder_var_figures+'Fc24h_'+station+'.png'
        plot_forecasts_24h_individual_operational(station,df_forecasts,pm2p5,\
            path_individual_figure_var,probabilities=True,df_probs=members_df)
        dic_forecasts[station] = df_forecasts
        dic_probs_dataframes[station] = members_df
        dic_pm2p5[station] = pm2p5
        dic_forecasts[station]['MEAN'] = dic_forecasts[station].mean(axis=1)
    except Exception:
        traceback.print_exc()

### Summary ###
path_summary = '/var/data1/AQ_Forecast_DATA/results/summary/'
path_figure_summary = path_summary+'png/'+'Fc24h_Summary_'+str_initial_date+'.png'
path_figure_summary_var = folder_var_figures+'Fc24h_Summary.png'
plot_summary_24h(dic_forecasts,dic_pm2p5,path_figure_summary)
plot_summary_24h(dic_forecasts,dic_pm2p5,path_figure_summary_var)

### Plot and save CAMS,IFRP,BackTrajectories ###
gfs_path = '/var/data1/AQ_Forecast_DATA/operational/GFS/'
cams_path = '/var/data1/AQ_Forecast_DATA/historic/CAMS/Pronostico/'
path_IFRP ="/var/data1/AQ_Forecast_DATA/historic/GFS/historic/Vientos/BT/IFRP/"
files_ifrp = np.sort(glob.glob(path_IFRP+'*'))
dates_ifrp = np.array([dt.datetime.strptime(files_ifrp[i].split('/')[-1][:-4],'%Y%m%d%H')\
    for i in range(len(files_ifrp))])

### Get GFS and CAMS ###
df_GFS,df_optimal_gfs = get_gfs_for_forecast(forecast_initial_date,\
    gfs_path,operational = False,latlon = None)
df_CAMS,df_optimal_cams = get_cams_for_forecast(forecast_initial_date,\
    cams_path,operational = False,latlon = None)

### Get IFRP
file_ifrp = files_ifrp[dates_ifrp<=forecast_initial_date+\
           dt.timedelta(hours=5)-dt.timedelta(hours=5)][-1]
df_IFRP = pd.read_csv(file_ifrp,index_col=1,parse_dates=True)[['IFRP']]
df_IFRP.index = df_IFRP.index-dt.timedelta(hours=5)

### Plot CAMS
path_output_cams = '/var/data1/AQ_Forecast_DATA/results/cams/png/FcAOD_'+str_initial_date+'.png'
path_output_cams_csv = '/var/data1/AQ_Forecast_DATA/results/cams/csv/FcAOD_'+str_initial_date+'.csv'
plot_cams_operacional(df_CAMS,index_future,path_output_cams,path_output_cams_csv)

path_output_cams_var = '/var/data1/AQ_Forecast_DATA/results/var/png/FcAOD.png'
path_output_cams_var_csv = '/var/data1/AQ_Forecast_DATA/results/var/csv/FcAOD.csv'
plot_cams_operacional(df_CAMS,index_future,path_output_cams_var,path_output_cams_var_csv)

### Plot IFRP
path_output_ifrp = '/var/data1/AQ_Forecast_DATA/results/ifrp/png/IFRP_'+str_initial_date+'.png'
plot_ifrp_forecast(df_IFRP,index_future,path_output_ifrp)
path_output_ifrp_var = '/var/data1/AQ_Forecast_DATA/results/var/png/IFRP.png'
plot_ifrp_forecast(df_IFRP,index_future,path_output_ifrp_var)

### Plot Trajectories
path_bt = '/var/data1/AQ_Forecast_DATA/historic/GFS/historic/Vientos/BT/'+\
    'BT_GFS.1h.800hPa.'+file_ifrp.split('/')[-1][:-4]+'.4days.nc'
path_fires = "/var/data1/AQ_Forecast_DATA/operational/Fires/MODIS_C6_1_South_America_7d.csv"

path_plot_bt = '/var/data1/AQ_Forecast_DATA/results/ifrp/png/Traj_'+str_initial_date+'.png'
plot_trajectories_hotspots(path_bt,path_fires,forecast_initial_date,path_plot_bt)
path_plot_bt_var = '/var/data1/AQ_Forecast_DATA/results/var/png/Traj.png'
plot_trajectories_hotspots(path_bt,path_fires,forecast_initial_date,path_plot_bt_var)


### Plot ICA figure (in test)
path_plot_ica = '/var/data1/AQ_Forecast_DATA/results/summary/png/Fc24h_ICAsummary_'+str_initial_date+'.png'
path_plot_ica_var = '/var/data1/AQ_Forecast_DATA/results/var/png/Fc24h_ICAsummary.png'
plot_ForeICA24h(forecast_initial_date, path_plot_ica)
plot_ForeICA24h(forecast_initial_date, path_plot_ica_var)


### Plot ICA figure - poblacionales (in test)
path_plot_ica = '/var/data1/AQ_Forecast_DATA/results/summary/png/Fc24h_ICAsummary_pob_'+str_initial_date+'.png'
path_plot_ica_var = '/var/data1/AQ_Forecast_DATA/results/var/png/Fc24h_ICAsummary_pob.png'
plot_ForeICA24h_pob(forecast_initial_date, path_plot_ica)
plot_ForeICA24h_pob(forecast_initial_date, path_plot_ica_var)


# Copiar todo a SAL:
sal_path = '/var/www/CalidadAire/Pronostico_PM25/'
os.system('scp -r /var/data1/AQ_Forecast_DATA/results/var/* calidadaire@siata.gov.co:'+\
    sal_path)