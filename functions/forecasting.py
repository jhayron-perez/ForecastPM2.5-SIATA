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

def get_optimal_window(variable,hour,gfs_correlations_path):
    return np.argmax(abs(np.load(gfs_correlations_path+'/'+variable+'.npy')[:,hour]))+1

def get_gfs_for_forecast(forecast_initial_date,path_data,\
    gfs_correlations_path = '/home/jsperezc/jupyter/AQ_Forecast/notebooks/ForecastDevelopment/Data/CorrsAnomsGFS/',\
    operational = True, latlon = None):
    ## forecast_date_0: initial hour of forecast (last hour in which pm2.5 data is available)
    ## path_data: folder with this type of files: gfs_0p25_2021050112.nc
    ## gfs_correlations_path: path with pm2.5 vs gfs variables correlation matrices (for optimally averaging)
    ## operational: False when it is a historic run
    ## latlon: coordenates of station, if None the output will be the mean for the Aburra Valley
    
    ### We need to get data for the next 5 days and the past day, then optimally average them
    gfs_files_names = np.sort(glob.glob(path_data+'gfs_0p25_*.nc'))
    initial_dates_gfs_files = np.array([dt.datetime.strptime(gfs_files_names[i].split('_')[-1].split('.')[0],\
        '%Y%m%d%H') for i in range(len(gfs_files_names))])
#     print(initial_dates_gfs_files)
    ### For operational run one simply takes the last two available files, for historic run we need to know
    ### the real delay of data (8 hours)
    
    latency = 8 # hours
    if operational == True:
        gfs_2_files = gfs_files_names[-2:]
    else:
        gfs_2_files = gfs_files_names[np.where(initial_dates_gfs_files - dt.timedelta(hours = 5) \
            <= forecast_initial_date - dt.timedelta(hours = latency))[0]][-2:]
    
    ### Read the files and keep the most updated value
    dataset_1 = xr.open_mfdataset(gfs_2_files[0])
    dataset_2 = xr.open_mfdataset(gfs_2_files[1])
    gfs_total = xr.concat([dataset_1,dataset_2], dim = "time")
    gfs_recorte = postprocessing.recorte_espacial(gfs_total)
    
    lat = gfs_recorte.latitude.values
    lon = gfs_recorte.longitude.values
    tcc = gfs_recorte.tcc_atm_avg.values
    rad = gfs_recorte.rad_in.values
    prate = gfs_recorte.prate_srf_avg.values
    hpbl = gfs_recorte.hpbl.values
    cin = gfs_recorte.cin.values
    
    if latlon == None: ## Get mean series
        mean_tcc = np.mean(gfs_recorte.tcc_atm_avg[:],axis = [1,2]).values #%
        mean_rad = np.mean(gfs_recorte.rad_in[:],axis = [1,2]).values #W/m2
        mean_prate = np.mean(gfs_recorte.prate_srf_avg[:],axis = [1,2]).values #kg m*-2 s*-1
        mean_hpbl = np.mean(gfs_recorte.hpbl[:],axis = [1,2]).values #m
        mean_cin = np.mean(gfs_recorte.cin[:],axis = [1,2]).values #J kg**-1

        df_GFS = pd.DataFrame(np.array([mean_tcc,mean_rad,mean_prate,mean_hpbl,mean_cin]).T,
                             index = gfs_recorte.time,columns = ['tcc','rad','prate','hpbl','cin'])
        df_GFS['prate'] = df_GFS['prate']*60*60
        df_GFS_temp = df_GFS.reset_index()
        df_GFS = df_GFS_temp.drop_duplicates("index",keep= "last") ## Esto para eliminar los datos que se solapan entre las diferentes corridas
        ## y conservar el último, que sería el más actualizado (última descarga)
        df_GFS.set_index("index", inplace=True)
        df_GFS.index = df_GFS.index-dt.timedelta(hours = 1.5)
        df_GFS_hourly = df_GFS.resample('H').mean().interpolate(method = 'linear',order = 3)
        df_GFS_hourly.index = df_GFS_hourly.index - dt.timedelta(hours = 5)

    else: ## Nearest point
        lat_station = latlon[0]
        lon_station = latlon[1]
        ilon = np.argmin(abs(lon_station-lon))
        ilat = np.argmin(abs(lat_station-lat))
        tcc_temp = tcc[:,ilat,ilon] #%
        rad_temp = rad[:,ilat,ilon] #W/m2
        prate_temp = prate[:,ilat,ilon] #kg m*-2 s*-1
        hpbl_temp = hpbl[:,ilat,ilon] #m
        cin_temp = cin[:,ilat,ilon] #J kg**-1
        df_GFS = pd.DataFrame(np.array([tcc_temp,rad_temp,prate_temp,hpbl_temp,cin_temp]).T,
                             index = gfs_recorte.time,columns = ['tcc','rad','prate','hpbl','cin'])
        df_GFS['prate'] = df_GFS['prate']*60*60
        df_GFS_temp = df_GFS.reset_index()
        df_GFS = df_GFS_temp.drop_duplicates("index",keep= "last") ## Esto para eliminar los datos que se solapan entre las diferentes corridas
        ## y conservar el último, que sería el más actualizado (última descarga)
        df_GFS.set_index("index", inplace=True)
        df_GFS.index = df_GFS.index-dt.timedelta(hours = 1.5)
        df_GFS_hourly = df_GFS.resample('H').mean().interpolate(method = 'linear',order = 3)
        df_GFS_hourly.index = df_GFS_hourly.index - dt.timedelta(hours = 5)
    
    keys_gfs = np.array(df_GFS_hourly.keys()).astype(str)
    
    ### Now get optimal average
    df_optimal = pd.DataFrame(index = df_GFS_hourly.index)
    for variable in keys_gfs:
        df_optimal_temp = copy.deepcopy(df_GFS_hourly[[variable]])
        for hour in range(0,24):
            optimal_window = get_optimal_window(variable,hour,gfs_correlations_path)
            df_optimal_temp[df_optimal_temp.index.hour == hour] = df_GFS_hourly[[variable]].rolling(optimal_window,min_periods=1).mean()\
                [df_optimal_temp.index.hour == hour]
        df_optimal[variable] = df_optimal_temp
        
    return df_GFS_hourly,df_optimal

def get_cams_for_forecast(forecast_initial_date,path_data,operational = True,latlon=None):
    ## forecast_date_0: initial hour of forecast (last hour in which pm2.5 data is available)
    ## path_data: if historic run then folder cams forecast subdirectory: year/month/CAMS_date_hour.nc
    ## gfs_correlations_path: path with pm2.5 vs gfs variables correlation matrices (for optimally averaging)
    ## operational: False when it is a historic run
    ## latlon: coordenates of station, if None the output will be the mean for the Aburra Valley
    
    latency = 10 # hours
    
    if operational == False:
        files_cams = []
        dirs_cams = []
        for root, dirs, files in os.walk(path_data,followlinks=True):
            for file in files:
                files_cams.append(file)
            for dirtemp in dirs:
                dirs_cams.append(dirtemp)
        files_cams = np.sort(files_cams)
        initial_dates_cams_files = np.array([dt.datetime.strptime(files_cams[i].split('/')[-1].split('.')[0],\
            'CAMS_%Y%m%d_%H%M%S') for i in range(len(files_cams))])
        cams_2_files = [files_cams[np.where(initial_dates_cams_files - dt.timedelta(hours = 5) \
            <= forecast_initial_date - dt.timedelta(hours = latency))[0]][-1]]
        tempfile1 = path_data+cams_2_files[0].split('_')[1][:4]+'/'+cams_2_files[0].split('_')[1][4:6]+'/'+cams_2_files[0]
        cams_2_files = [tempfile1]
    else:
        files_cams = np.sort(glob.glob(path_data+'CAMS_*_*.nc'))
        initial_dates_cams_files = np.array([dt.datetime.strptime(files_cams[i].split('/')[-1].split('.')[0],\
            'CAMS_%Y%m%d_%H%M%S') for i in range(len(files_cams))])
        cams_2_files = files_cams[-1]
    ### Read the files and keep the most updated value
    dataset_1 = xr.open_mfdataset([cams_2_files])
    cams_total = xr.concat([dataset_1], dim = "time")
    cams_recorte = postprocessing.recorte_espacial(cams_total)
    variables_cams = ['pm2p5','aod550','bcaod550','duaod550','omaod550','ssaod550','niaod550','amaod550','suaod550']
    
    lat = cams_recorte.latitude.values
    lon = cams_recorte.longitude.values

    if latlon == None: ## Get mean series
        dic_cams = {}
        for variable in variables_cams:
            dic_cams[variable] = np.mean(cams_recorte[variable][:],axis = [1,2]).values
        df_CAMS = pd.DataFrame(dic_cams,index = cams_recorte.time)
        df_CAMS['pm2p5'] = df_CAMS['pm2p5'] * 1000_000_000 #ug/m3
    else: ## Nearest point
        lat_station = latlon[0]
        lon_station = latlon[1]
        ilon = np.argmin(abs(lon_station-lon))
        ilat = np.argmin(abs(lat_station-lat))

        dic_cams = {}
        for variable in variables_cams:
            dic_cams[variable] = cams_recorte[variable][:,ilat,ilon].values
        df_CAMS = pd.DataFrame(dic_cams,index = cams_recorte.time)
        df_CAMS['pm2p5'] = df_CAMS['pm2p5'] * 1000_000_000 #ug/m3
    
    df_CAMS.index = df_CAMS.index - dt.timedelta(hours = 5)

    df_CAMS = df_CAMS[['pm2p5','aod550','bcaod550','duaod550','omaod550','ssaod550','niaod550','amaod550','suaod550']]
    df_CAMS.columns = ['pm2p5_cams','aod','bcaod','duaod','omaod','ssaod','niaod','amaod','suaod']
    print(df_CAMS)
    df_optimal_cams = pd.DataFrame(index = df_CAMS.index)
    df_optimal_cams['pm2p5_cams'] = df_CAMS.pm2p5_cams.rolling(3,min_periods=1).mean()
    df_optimal_cams['aod'] = df_CAMS.aod.rolling(3,min_periods=1).mean()
#     df_optimal_cams['ssaod'] = df_CAMS.ssaod.rolling(9,min_periods=1).mean()
#     df_optimal_cams['omaod'] = df_CAMS.omaod.rolling(12,min_periods=1).mean()
#     df_optimal_cams['duaod'] = df_CAMS.duaod.rolling(1,min_periods=1).mean()
    df_optimal_cams['bcaod'] = df_CAMS.duaod.rolling(1,min_periods=1).mean()
    
    return df_CAMS,df_optimal_cams

def get_pm2p5_for_validation(forecast_initial_date,station_name=None):
    df_pm2p5 = read.PM25(str(forecast_initial_date+dt.timedelta(hours = 2)),\
                         str(forecast_initial_date+dt.timedelta(hours = 97))).data
    trained_stations = ['ITA-CJUS','ITA-CONC','MED-LAYE','CAL-JOAR','EST-HOSP','MED-ALTA','MED-VILL',
                        'BAR-TORR','COP-CVID','MED-BEME','MED-TESO','MED-SCRI','MED-ARAN','BEL-FEVE','ENV-HOSP',
                        'SAB-RAME','MED-SELE']
    df_pm2p5 = df_pm2p5[trained_stations]
#     print(df_pm2p5)
    df_pm2p5.index = df_pm2p5.index#-dt.timedelta(hours = 1)
    if station_name==None:
        df_pm2p5 = pd.DataFrame(df_pm2p5.mean(axis=1),columns=['PM2.5'])
        df_pm2p5 = df_pm2p5.interpolate(limit=12)
    else:
        df_pm2p5 = pd.DataFrame(df_pm2p5[station_name].values,index=df_pm2p5.index,columns=['PM2.5'])
        df_pm2p5 = df_pm2p5.interpolate(limit=12)
    return df_pm2p5

### get full pm2.5
def get_pm2p5_period(initial_date,final_date,station_name=None):
    df_pm2p5 = read.PM25(str(initial_date),\
                         str(final_date)).data
    trained_stations = ['ITA-CJUS','ITA-CONC','MED-LAYE','CAL-JOAR','EST-HOSP','MED-ALTA','MED-VILL',
                        'BAR-TORR','COP-CVID','MED-BEME','MED-TESO','MED-SCRI','MED-ARAN','BEL-FEVE','ENV-HOSP',
                        'SAB-RAME','MED-SELE']
    df_pm2p5 = df_pm2p5[trained_stations]
#     print(df_pm2p5)
    df_pm2p5.index = df_pm2p5.index#-dt.timedelta(hours = 1)
    if station_name==None:
        df_pm2p5 = pd.DataFrame(df_pm2p5.mean(axis=1),columns=['PM2.5'])
        df_pm2p5 = df_pm2p5.interpolate(limit=12)
    else:
        df_pm2p5 = pd.DataFrame(df_pm2p5[station_name].values,index=df_pm2p5.index,columns=['PM2.5'])
        df_pm2p5 = df_pm2p5.interpolate(limit=12)
    return df_pm2p5


def forecast_pm2p5(forecast_initial_date,df_pm2p5,df_optimal_cams,df_optimal_gfs,station_name=None, model_name = 'KNN',\
    mos_correction = False):
    if station_name == None:
        folder_models = '/var/data1/AQ_Forecast_DATA/trained_models/final_models/'
    else:
        folder_models = '/var/data1/AQ_Forecast_DATA/trained_models/final_models_stations/'
        
    ### 26 last hours of pm2p5

    ### Dataframe with lagged data
    max_lag = 24 ## Maximum lag used
    df_add = pd.DataFrame([[np.nan]], columns=['PM2.5'],index=[forecast_initial_date])
    df_pm2p5 = df_pm2p5.append(df_add)
#    print(df_pm2p5)
    x_shifts = pd.concat([df_pm2p5['PM2.5'].shift(i) for i in range(max_lag,0,-1)],axis = 1)#.dropna()
    x_shifts.columns = (-np.arange(max_lag,0,-1)).astype(str)
 #   print(x_shifts)
    
    #CAMS
    x_cams = df_optimal_cams[['pm2p5_cams','aod','ssaod','omaod','duaod','bcaod']].loc\
        [forecast_initial_date:\
        forecast_initial_date+dt.timedelta(hours = 95)].values

    #GFS
    x_gfs = df_optimal_gfs[['tcc','rad','prate','hpbl','cin']].loc\
        [forecast_initial_date:\
        forecast_initial_date+dt.timedelta(hours = 95)].values
    
    index_future = pd.date_range(forecast_initial_date,\
                  forecast_initial_date+dt.timedelta(hours = 95),freq='H')
    
    x_hour = index_future.hour.values.astype(float)
#     print(x_hour)
    x_dow = index_future.day_of_week.values.astype(float)
    if station_name == None:
        scaler_x = pickle.load(open(folder_models+'scaler_x.scl', 'rb'))
        scaler_y = pickle.load(open(folder_models+'scaler_y.scl', 'rb'))
    else:
        scaler_x = pickle.load(open(folder_models+station_name+'scaler_x.scl', 'rb'))
        scaler_y = pickle.load(open(folder_models+station_name+'scaler_y.scl', 'rb'))
    pm_lagged_container = copy.deepcopy(x_shifts).loc[forecast_initial_date].values.T

    y_predictions = []
#     print(model_name)
    for leadtime in range(1,97):
        if station_name == None:
            filename = folder_models + model_name + '.mdl'
        else:
            filename = folder_models + station_name + '_'+ model_name + '.mdl'
        estimator = pickle.load(open(filename, 'rb'))

        X = np.hstack([pm_lagged_container,x_cams[leadtime-1],\
                       x_gfs[leadtime-1],x_hour[leadtime-1],\
                       x_dow[leadtime-1]])
        X = scaler_x.transform(X.reshape(1, -1))
        if mos_correction == False:
            y_predicted = scaler_y.inverse_transform(estimator.predict(X).reshape(1, -1))
            y_predictions.append(y_predicted[0])
            pm_lagged_container = np.hstack([pm_lagged_container,y_predicted[0]])[1:]
        else:
            if station_name == None:
                folder_mos = '/var/data1/AQ_Forecast_DATA/trained_models/mos_operators/'
            else:
                folder_mos = folder_models + 'mos_operators/' + station_name + '/'
            with open(folder_mos+model_name+'lt_'+str(leadtime).zfill(2)+'.mos','rb') as f:
                mos_temp = pickle.load(f)
            y_predicted = scaler_y.inverse_transform(estimator.predict(X).reshape(1, -1))
            pm_lagged_container = np.hstack([pm_lagged_container,y_predicted[0]])[1:]
            y_predicted = mos_temp(y_predicted)
            y_predictions.append(y_predicted[0])
#             print(y_predicted)


    y_predictions = pd.DataFrame(y_predictions,index= index_future)
    return y_predictions

def forecast_pm2p5_v2(forecast_initial_date,df_pm2p5,df_optimal_cams,df_optimal_gfs,station_name=None, model_name = 'KNN',\
    mos_correction = False):
    if station_name == None:
        folder_models = '/var/data1/AQ_Forecast_DATA/trained_models/final_models/'
    else:
        folder_models = '/var/data1/AQ_Forecast_DATA/trained_models/final_models_stations/'
        
    ### 26 last hours of pm2p5

    ### Dataframe with lagged data
    max_lag = 24 ## Maximum lag used
    df_add = pd.DataFrame([[np.nan]], columns=['PM2.5'],index=[forecast_initial_date])
    df_pm2p5 = df_pm2p5.append(df_add)
#    print(df_pm2p5)
    x_shifts = pd.concat([df_pm2p5['PM2.5'].shift(i) for i in range(max_lag,0,-1)],axis = 1)#.dropna()
    x_shifts.columns = (-np.arange(max_lag,0,-1)).astype(str)
 #   print(x_shifts)
    
    #CAMS
    x_cams = df_optimal_cams[['pm2p5_cams','aod','ssaod','omaod','duaod','bcaod']].loc\
        [forecast_initial_date:\
        forecast_initial_date+dt.timedelta(hours = 95)].values

    #GFS
    x_gfs = df_optimal_gfs[['tcc','rad','prate','hpbl','cin']].loc\
        [forecast_initial_date:\
        forecast_initial_date+dt.timedelta(hours = 95)].values
    
    index_future = pd.date_range(forecast_initial_date,\
                  forecast_initial_date+dt.timedelta(hours = 95),freq='H')
    
    x_hour = index_future.hour.values.astype(float)
#     print(x_hour)
    x_dow = index_future.day_of_week.values.astype(float)
    df_ifrp = pd.read_csv('/var/data1/AQ_Forecast_DATA/operational/Fires/IFRP.csv',index_col = 0,parse_dates = True)
    x_ifrp = np.repeat(df_ifrp.iloc[-1],96)
    
    if station_name == None:
        scaler_x = pickle.load(open(folder_models+'scaler_x_v2.scl', 'rb'))
        scaler_y = pickle.load(open(folder_models+'scaler_y_v2.scl', 'rb'))
    else:
        scaler_x = pickle.load(open(folder_models+station_name+'scaler_x_v2.scl', 'rb'))
        scaler_y = pickle.load(open(folder_models+station_name+'scaler_y_v2.scl', 'rb'))
    pm_lagged_container = copy.deepcopy(x_shifts).loc[forecast_initial_date].values.T

    y_predictions = []
#     print(model_name)
    for leadtime in range(1,97):
        if station_name == None:
            filename = folder_models + model_name + '_v2.mdl'
        else:
            filename = folder_models + station_name + '_'+ model_name + '_v2.mdl'
        estimator = pickle.load(open(filename, 'rb'))

        X = np.hstack([pm_lagged_container,x_cams[leadtime-1],\
                       x_gfs[leadtime-1],x_hour[leadtime-1],\
                       x_dow[leadtime-1],x_ifrp[leadtime-1]])
        X = scaler_x.transform(X.reshape(1, -1))
#         print(X.shape)
        if mos_correction == False:
            y_predicted = scaler_y.inverse_transform(estimator.predict(X).reshape(1, -1))
            y_predictions.append(y_predicted[0])
            pm_lagged_container = np.hstack([pm_lagged_container,y_predicted[0]])[1:]
        else:
            if station_name == None:
                folder_mos = '/var/data1/AQ_Forecast_DATA/trained_models/mos_operators/'
            else:
                folder_mos = folder_models + 'mos_operators/' + station_name + '/'
            with open(folder_mos+model_name+'lt_'+str(leadtime).zfill(2)+'_v2.mos','rb') as f:
                mos_temp = pickle.load(f)
            y_predicted = scaler_y.inverse_transform(estimator.predict(X).reshape(1, -1))
            pm_lagged_container = np.hstack([pm_lagged_container,y_predicted[0]])[1:]
            y_predicted = mos_temp(y_predicted)
            y_predictions.append(y_predicted[0])
#             print(y_predicted)


    y_predictions = pd.DataFrame(y_predictions,index= index_future)
    return y_predictions


def forecast_pm2p5_v3(forecast_initial_date,df_pm2p5,df_optimal_cams,df_optimal_gfs,station_name=None, model_name = 'KNN',\
    mos_correction = False):
    if station_name == None:
        folder_models = '/var/data1/AQ_Forecast_DATA/trained_models/final_models/'
    else:
        folder_models = '/var/data1/AQ_Forecast_DATA/trained_models/final_models_stations/'
        
    ### 26 last hours of pm2p5

    ### Dataframe with lagged data
    max_lag = 11 ## Maximum lag used
    df_add = pd.DataFrame([[np.nan]], columns=['PM2.5'],index=[forecast_initial_date])
    df_pm2p5 = df_pm2p5.append(df_add)
#    print(df_pm2p5)
    x_shifts = pd.concat([df_pm2p5['PM2.5'].shift(i) for i in range(max_lag,0,-1)],axis = 1)#.dropna()
    x_shifts.columns = (-np.arange(max_lag,0,-1)).astype(str)
 #   print(x_shifts)
    
    #CAMS
    x_cams = df_optimal_cams[['aod','bcaod','pm2p5_cams']].loc\
        [forecast_initial_date:\
        forecast_initial_date+dt.timedelta(hours = 95)].values

    #GFS
    x_gfs = df_optimal_gfs[['tcc','rad','prate','hpbl','cin']].loc\
        [forecast_initial_date:\
        forecast_initial_date+dt.timedelta(hours = 95)].values
    
    index_future = pd.date_range(forecast_initial_date,\
                  forecast_initial_date+dt.timedelta(hours = 95),freq='H')
    
    x_hour = index_future.hour.values.astype(float)
#     print(x_hour)
    x_dow = index_future.day_of_week.values.astype(float)
    df_ifrp = pd.read_csv('/var/data1/AQ_Forecast_DATA/operational/Fires/IFRP.csv',index_col = 0,parse_dates = True)
    x_ifrp = np.repeat(df_ifrp.iloc[-1],96)
    
    if station_name == None:
        scaler_x = pickle.load(open(folder_models+'scaler_x_v3.scl', 'rb'))
        scaler_y = pickle.load(open(folder_models+'scaler_y_v3.scl', 'rb'))
    else:
        scaler_x = pickle.load(open(folder_models+station_name+'scaler_x_v3.scl', 'rb'))
        scaler_y = pickle.load(open(folder_models+station_name+'scaler_y_v3.scl', 'rb'))
    pm_lagged_container = copy.deepcopy(x_shifts).loc[forecast_initial_date].values.T

    y_predictions = []
#     print(model_name)
    for leadtime in range(1,97):
        if station_name == None:
            filename = folder_models + model_name + '_v3.mdl'
        else:
            filename = folder_models + station_name + '_'+ model_name + '_v3.mdl'
        estimator = pickle.load(open(filename, 'rb'))

        X = np.hstack([pm_lagged_container,x_cams[leadtime-1],\
                       x_gfs[leadtime-1],x_hour[leadtime-1],\
                       x_dow[leadtime-1],x_ifrp[leadtime-1]])
        X = scaler_x.transform(X.reshape(1, -1))
#         print(X.shape)
        if mos_correction == False:
            y_predicted = scaler_y.inverse_transform(estimator.predict(X).reshape(1, -1))
            y_predictions.append(y_predicted[0])
            pm_lagged_container = np.hstack([pm_lagged_container,y_predicted[0]])[1:]
        else:
            if station_name == None:
                folder_mos = '/var/data1/AQ_Forecast_DATA/trained_models/mos_operators/'
            else:
                folder_mos = folder_models + 'mos_operators/' + station_name + '/'
            with open(folder_mos+model_name+'lt_'+str(leadtime).zfill(2)+'_v2.mos','rb') as f:
                mos_temp = pickle.load(f)
            y_predicted = scaler_y.inverse_transform(estimator.predict(X).reshape(1, -1))
            pm_lagged_container = np.hstack([pm_lagged_container,y_predicted[0]])[1:]
            y_predicted = mos_temp(y_predicted)
            y_predictions.append(y_predicted[0])
#             print(y_predicted)


    y_predictions = pd.DataFrame(y_predictions,index= index_future)
    return y_predictions

def forecast_pm2p5_v4(forecast_initial_date,df_pm2p5,df_optimal_cams,df_optimal_gfs,station_name=None, model_name = 'KNN',\
    mos_correction = False):
#     if station_name == None:
    folder_models = '/var/data1/AQ_Forecast_DATA/trained_models/final_models/'
#     else:
    folder_models_stations = '/var/data1/AQ_Forecast_DATA/trained_models/final_models_stations/'
        
    ### 26 last hours of pm2p5

    ### Dataframe with lagged data
    max_lag = 48 ## Maximum lag used
    df_add = pd.DataFrame([[np.nan]], columns=['PM2.5'],index=[forecast_initial_date])
    df_pm2p5 = df_pm2p5.append(df_add)
#    print(df_pm2p5)
    x_shifts = pd.concat([df_pm2p5['PM2.5'].shift(i) for i in range(max_lag,0,-1)],axis = 1)#.dropna()
    x_shifts.columns = (-np.arange(max_lag,0,-1)).astype(str)
 #   print(x_shifts)
    
    #CAMS
    x_cams = df_optimal_cams[['aod','bcaod']].loc\
        [forecast_initial_date:\
        forecast_initial_date+dt.timedelta(hours = 95)].values
    
    #GFS
    x_gfs = df_optimal_gfs[['tcc','rad','prate','hpbl','cin']].loc\
        [forecast_initial_date:\
        forecast_initial_date+dt.timedelta(hours = 95)].values
    print('hola')
    print(df_optimal_gfs)
    print(x_gfs)
    index_future = pd.date_range(forecast_initial_date,\
                  forecast_initial_date+dt.timedelta(hours = 95),freq='H')

    x_hour = index_future.hour.values.astype(float)
#     print(x_hour)
    x_dow = index_future.day_of_week.values.astype(float)
    files_ifrp = np.sort(glob.glob('/var/data1/AQ_Forecast_DATA/historic/GFS/historic/Vientos/BT/IFRP/*.csv'))
    last_file = files_ifrp[-1]
    df_ifrp = pd.read_csv(last_file,index_col = 1,parse_dates = True)['IFRP'].rolling(24,min_periods=0).mean()
    df_ifrp.index = df_ifrp.index-dt.timedelta(hours = 5)

    x_ifrp = df_ifrp.loc\
        [forecast_initial_date:\
        forecast_initial_date+dt.timedelta(hours = 95)]#.values
#     print(x_ifrp)
#     aaaa
    print(len(x_cams),len(x_gfs),len(x_hour),len(x_dow),len(x_ifrp))
    if station_name == None:
        scaler_x = pickle.load(open(folder_models+'scaler_x_v4.scl', 'rb'))
        scaler_y = pickle.load(open(folder_models+'scaler_y_v4.scl', 'rb'))
    else:
        scaler_x = pickle.load(open(folder_models_stations+station_name+'scaler_x_v4.scl', 'rb'))
        scaler_y = pickle.load(open(folder_models_stations+station_name+'scaler_y_v4.scl', 'rb'))
    pm_lagged_container = copy.deepcopy(x_shifts).loc[forecast_initial_date].values.T

    y_predictions = []
#     print(model_name)
    for leadtime in range(1,97):
#         if station_name == None:
        filename = folder_models + model_name + '_v4.mdl'
#         else:
#             filename = folder_models + station_name + '_'+ model_name + '_v4.mdl'
        estimator = pickle.load(open(filename, 'rb'))

        X = np.hstack([pm_lagged_container,x_cams[leadtime-1],\
                       x_gfs[leadtime-1],x_hour[leadtime-1],\
                       x_dow[leadtime-1],x_ifrp[leadtime-1]])
        X = scaler_x.transform(X.reshape(1, -1))
#         print(X.shape)
        if mos_correction == False:
            y_predicted = scaler_y.inverse_transform(estimator.predict(X).reshape(1, -1))
            y_predictions.append(y_predicted[0])
            pm_lagged_container = np.hstack([pm_lagged_container,y_predicted[0]])[1:]
        else:
            if station_name == None:
                folder_mos = '/var/data1/AQ_Forecast_DATA/trained_models/mos_operators/'
            else:
                folder_mos = folder_models + 'mos_operators/' + station_name + '/'
            with open(folder_mos+model_name+'lt_'+str(leadtime).zfill(2)+'_v2.mos','rb') as f:
                mos_temp = pickle.load(f)
            y_predicted = scaler_y.inverse_transform(estimator.predict(X).reshape(1, -1))
            pm_lagged_container = np.hstack([pm_lagged_container,y_predicted[0]])[1:]
            y_predicted = mos_temp(y_predicted)
            y_predictions.append(y_predicted[0])
#             print(y_predicted)


    y_predictions = pd.DataFrame(y_predictions,index= index_future)
    return y_predictions

def predict_probabilities_mo(station, x, index_future, members=50):
    import random
    import joblib
    
    model_name = 'RF_MO'
    estimator = joblib.load('/var/data1/AQ_Forecast_DATA/trained_estimators/'+station+'_'+model_name+'.mdl')
    scaler_x = joblib.load('/var/data1/AQ_Forecast_DATA/scalers/'+station+'_X.scl')
    scaler_y = joblib.load('/var/data1/AQ_Forecast_DATA/scalers/'+station+'_Y.scl')
    x_scaled = scaler_x.transform(x.reshape(1, -1))
    
    Y_scaled = np.empty([members, len(estimator.estimators_)])
    for i, clf in enumerate(estimator.estimators_):
        estimators_temp = random.choices(clf.estimators_,k=members)
        for j, estimator_temp in enumerate(estimators_temp):
            Y_scaled[j,i] = estimator_temp.predict(np.hstack([x_scaled])) 
    Y = scaler_y.inverse_transform(Y_scaled)
    
    ### Compute percentiles ###
    df_probs = pd.DataFrame(index=index_future)
    df_probs['max'] = np.nanmax(Y,axis=0)
    df_probs['min'] = np.nanmin(Y,axis=0)
    df_probs['p5'] = np.nanpercentile(Y,5,axis=0)
    df_probs['p10'] = np.nanpercentile(Y,10,axis=0)
    df_probs['p25'] = np.nanpercentile(Y,25,axis=0)
    df_probs['p50'] = np.nanpercentile(Y,50,axis=0)
    df_probs['p75'] = np.nanpercentile(Y,75,axis=0)
    df_probs['p90'] = np.nanpercentile(Y,90,axis=0)
    df_probs['p95'] = np.nanpercentile(Y,95,axis=0)
    
    return Y, df_probs

def forecast_station(x,index_future,station,models):
    import joblib
    df_results = pd.DataFrame(index=index_future)
    
    for model_name in models:
        estimator_temp = joblib.load('/var/data1/AQ_Forecast_DATA/trained_estimators/'+station+'_'+model_name+'.mdl')
        scaler_x = joblib.load('/var/data1/AQ_Forecast_DATA/scalers/'+station+'_X.scl')
        scaler_y = joblib.load('/var/data1/AQ_Forecast_DATA/scalers/'+station+'_Y.scl')
        x_scaled = scaler_x.transform(x.reshape(1, -1))
        prediction_scaled = estimator_temp.predict(x_scaled)
        prediction = scaler_y.inverse_transform(prediction_scaled)

        df_results[model_name] = prediction[0]
    return df_results
