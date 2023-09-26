import pandas as pd
import sys
sys.path.insert(1,'/home/jsperezc/jupyter/AQ_Forecast/functions/')
sys.path.append('/home/calidadaire/Paquete/')
import airquality.read_data as read
import numpy as np
import xarray as xr
import datetime as dt
import glob
import postprocessing
import copy
import os

def recorte_datos(path_files, coordenadas):
    """
    Esta funcion Permite recortar archivos NetCDF o GRIB haciendo uso de la libreria xarray.
    
    Para esto, debe ingresar la ruta del directorio donde estan los archivos a recortar, ademas de un vector con las coordenadas 
    en las siguientes posiciones: [N,S,E,O] 
    
    Tenga en cuenta que se guardara el nuevo archivo con el mismo nombre que el archivo original, es decir, se va a sobreescribir 
    este ultimo
    """
    import xarray as xr
    import os
    import glob
    
    Files = glob.glob(path_files+'*')
    Name_Files = os.listdir(path_files)
    for i,file in enumerate(Files):
        global_data = xr.open_dataset(file)
        lon = global_data.coords["lon"]
        lat = global_data.coords["lat"]

        #Hacemos el recorte de los datos a nuestra region de interés
        split_data = global_data.loc[dict(lon=lon[(lon > coordenadas[3]) & (lon < coordenadas[2])], 
                                          lat=lat[(lat > coordenadas[1]) & (lat < coordenadas[0])])] 

        #Guardamos el nuevo NetCDf
        split_data.to_netcdf(path_files+Name_Files[i], 
                                  unlimited_dims='time',format="NETCDF4_CLASSIC") 
        
def get_pm2p5_period(initial_date,final_date,station_name=None):

    df_pm2p5 = read.PM25(str(initial_date),\
                         str(final_date)).data
    trained_stations = ['ITA-CJUS','CAL-LASA','ITA-CONC','MED-LAYE','CAL-JOAR','EST-HOSP','MED-ALTA','MED-VILL',
                        'BAR-TORR','COP-CVID','MED-BEME','MED-TESO','MED-SCRI','MED-ARAN','BEL-FEVE','ENV-HOSP',
                        'SAB-RAME','MED-SELE']

    if station_name==None:
        df_pm2p5 = df_pm2p5[trained_stations]
        df_pm2p5 = pd.DataFrame(df_pm2p5.mean(axis=1),columns=['PM2.5'])
        df_pm2p5 = df_pm2p5.interpolate(limit=12)
    else:
        df_pm2p5 = pd.DataFrame(df_pm2p5[station_name].values,index=df_pm2p5.index,columns=['PM2.5'])
        df_pm2p5 = df_pm2p5.interpolate(limit=12)
    return df_pm2p5

def get_gfs_for_forecast(forecast_initial_date,path_data,\
    gfs_correlations_path = '/var/data1/AQ_Forecast_DATA/historic/GFS/correlations',\
    operational = True, latlon = None):
    ## forecast_date_0: initial hour of forecast (last hour in which pm2.5 data is available)
    ## path_data: folder with this type of files: gfs_0p25_2021050112.nc
    ## gfs_correlations_path: path with pm2.5 vs gfs variables correlation matrices (for optimally averaging)
    ## operational: False when it is a historic run
    ## latlon: coordenates of station, if None the output will be the mean for the Aburra Valley
    
    ### We need to get data for the next 5 days and the past day, then optimally average them
    gfs_files_names = np.sort(glob.glob(path_data+'gfs_0p25_*.nc'))
#     print(gfs_files_names)
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
#     print(cams_2_files)
    dataset_1 = xr.open_mfdataset(cams_2_files)
    cams_total = xr.concat([dataset_1], dim = "time")
    cams_recorte = postprocessing.recorte_espacial(cams_total)
    variables_cams = ['aod550','bcaod550','pm2p5','duaod550','omaod550','ssaod550','niaod550','amaod550','suaod550']
#     variables_cams = ['aod550','bcaod550','pm2p5']
    
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

#     df_CAMS = df_CAMS[['aod550','bcaod550','pm2p5']]
    df_CAMS = df_CAMS[['aod550','bcaod550','pm2p5','duaod550','omaod550','ssaod550','niaod550','amaod550','suaod550']]
    df_CAMS.columns = ['aod','bcaod','pm2p5_cams','duaod','omaod','ssaod','niaod','amaod','suaod']
#     df_CAMS.columns = ['aod','bcaod','pm2p5_cams']    
#     print(df_CAMS)
    df_optimal_cams = pd.DataFrame(index = df_CAMS.index)
    df_optimal_cams['aod'] = df_CAMS.aod.rolling(3,min_periods=1).mean()
#     df_optimal_cams['ssaod'] = df_CAMS.ssaod.rolling(9,min_periods=1).mean()
#     df_optimal_cams['omaod'] = df_CAMS.omaod.rolling(12,min_periods=1).mean()
#     df_optimal_cams['duaod'] = df_CAMS.duaod.rolling(1,min_periods=1).mean()
    df_optimal_cams['bcaod'] = df_CAMS.bcaod.rolling(1,min_periods=1).mean()
    df_optimal_cams['pm2p5_cams'] = df_CAMS.pm2p5_cams.rolling(3,min_periods=1).mean()
    
    return df_CAMS,df_optimal_cams

def get_optimal_window(variable,hour,gfs_correlations_path):
    return np.argmax(abs(np.load(gfs_correlations_path+'/CorrsAnomsGFS_'+variable+'_v3.npy')[:,hour]))+1

def get_pm2p5_for_forecast(forecast_initial_date,station_name=None):
    df_pm2p5 = read.PM25(str(forecast_initial_date-dt.timedelta(hours = 96)),\
                         str(forecast_initial_date)).data
    trained_stations = ["ITA-CJUS", "ITA-CONC", "MED-LAYE", "CAL-JOAR", "EST-HOSP", "MED-ALTA", "MED-VILL",
            "BAR-TORR", "COP-CVID", "MED-BEME", "MED-TESO", "MED-SCRI", "MED-ARAN", "BEL-FEVE", "ENV-HOSP", 
            "SAB-RAME", "MED-SELE","CEN-TRAF"]
    
    if station_name==None:
        df_pm2p5 = df_pm2p5[trained_stations]
        df_pm2p5 = pd.DataFrame(df_pm2p5.mean(axis=1),columns=['PM2.5'])
        df_pm2p5 = df_pm2p5.interpolate(limit=24)
    else:
        df_pm2p5 = pd.DataFrame(df_pm2p5[station_name].values,index=df_pm2p5.index,columns=['PM2.5'])
        df_pm2p5 = df_pm2p5.interpolate(limit=24)
    return df_pm2p5

def get_inputs_for_forecast(station, forecast_initial_date):
    gfs_path = '/var/data1/AQ_Forecast_DATA/operational/GFS/'
    cams_path = '/var/data1/AQ_Forecast_DATA/historic/CAMS/Pronostico/'
    path_IFRP ="/var/data1/AQ_Forecast_DATA/historic/GFS/historic/Vientos/BT/IFRP/"
    files_ifrp = np.sort(glob.glob(path_IFRP+'*'))
    dates_ifrp = np.array([dt.datetime.strptime(files_ifrp[i].split('/')[-1][:-4],'%Y%m%d%H')\
        for i in range(len(files_ifrp))])

    coor_esta= pd.read_csv("/var/data1/AQ_Forecast_DATA/historic/PM25/CoordenadasEstaciones.csv", 
                           index_col= "Nombre")
    lat_est = coor_esta.loc[station].Latitud
    lon_est = coor_esta.loc[station].Longitud

    # Get PM2.5

    pm2p5 = get_pm2p5_for_forecast(forecast_initial_date-dt.timedelta(hours=1),station_name=station)
    pm2p5.index = pm2p5.index+dt.timedelta(hours=1)

    ### Get GFS and CAMS ###
    df_GFS,df_optimal_gfs = get_gfs_for_forecast(forecast_initial_date,\
        gfs_path,operational = False,latlon = (lat_est,lon_est))
    df_CAMS,df_optimal_cams = get_cams_for_forecast(forecast_initial_date,\
        cams_path,operational = False,latlon = (lat_est,lon_est))

    ### Get IFRP
    file_ifrp = files_ifrp[dates_ifrp<=forecast_initial_date+\
               dt.timedelta(hours=5)-dt.timedelta(hours=5)][-1]
    df_IFRP = pd.read_csv(file_ifrp,index_col=1,parse_dates=True)[['IFRP']]
    df_IFRP.index = df_IFRP.index-dt.timedelta(hours=5)

    # Lagged data
    x_shifts = pm2p5.loc[forecast_initial_date-dt.timedelta(hours=47):forecast_initial_date].values[:,0]

    index_future = pd.date_range(forecast_initial_date+dt.timedelta(hours=1),\
                  forecast_initial_date+dt.timedelta(hours=96),freq='H')

    # CAMS
    cams_future = df_optimal_cams.loc[index_future[0]:].iloc[np.arange(0,97,3),:]
    aod_future = cams_future['aod'].values

    # GFS
    gfs_future = df_optimal_gfs.loc[index_future[0]:].iloc[np.arange(0,97,3),:]
    tcc_future = gfs_future['tcc'].values
    prate_future = gfs_future['prate'].values
    hpbl_future = gfs_future['hpbl'].values

    # IFRP
    ifrp50_future = df_IFRP.rolling(3,center=True).max().loc[index_future[0]:].iloc[np.arange(0,97,3),0].values

    # HOD/DOW
    df_hour = pd.DataFrame(index_future.hour, index = index_future)
    hod1_future = np.sin(2*np.pi*(df_hour/24)).iloc[np.arange(0,24,1),0].values
    hod2_future = np.cos(2*np.pi*(df_hour/24)).iloc[np.arange(0,24,1),0].values

    df_dow = pd.DataFrame(index_future.dayofweek, index = index_future)
    dow1_future = np.sin(2*np.pi*(df_dow/7)).iloc[np.arange(0,96,24),0].values
    dow2_future = np.cos(2*np.pi*(df_dow/7)).iloc[np.arange(0,96,24),0].values

    x_temp = np.hstack([x_shifts,
        aod_future,
        tcc_future,
        prate_future,
        hpbl_future,
        ifrp50_future,
        dow1_future,
        dow2_future,
        hod1_future,
        hod2_future])
    
    return index_future, x_temp, pm2p5, df_GFS, df_CAMS, df_IFRP



#### MODIFICACIONES A FUNCIONES ORIGINALES ####
def get_gfs_for_historic(forecast_initial_date,path_data,\
    gfs_correlations_path = '/var/data1/AQ_Forecast_DATA/historic/GFS/correlations',\
    latlon = None):
    ## forecast_date_0: initial hour of forecast (last hour in which pm2.5 data is available)
    ## path_data: folder with this type of files: gfs_0p25_2021050112.nc
    ## gfs_correlations_path: path with pm2.5 vs gfs variables correlation matrices (for optimally averaging)
    ## operational: False when it is a historic run
    ## latlon: coordenates of station, if None the output will be the mean for the Aburra Valley
    
    
    ### We need to get data for the next 5 days and the past day, then optimally average them
    #Para datos históricos (que se almacenan en el server)
    Ls_gfs_paths = []
    Ls_gfs_files_names = []

    for init_hour in ['00', '06', '12', '18']:
        #Obteniendo archivos en cada una de las carpetas
        path_data_hour = path_data+'{0}/'.format(init_hour)
        path_names = np.sort(glob.glob(path_data_hour+'gfs_0p25_*.nc'))
        files_names = pd.Series(path_names).str.split('/', expand=True).iloc[:, -1].values

        #Guardando
        Ls_gfs_paths.append(path_names)
        Ls_gfs_files_names.append(files_names)


    #Concatenando y ordenando
    gfs_files_names = np.sort(np.concatenate(Ls_gfs_files_names))

    #Definiendo fechas a partir de los archivos
    initial_dates_gfs_files = np.array([dt.datetime.strptime(gfs_files_names[i].split('_')[-1].split('.')[0],\
        '%Y%m%d%H') for i in range(len(gfs_files_names))])


    #Obteniendo archivos a leer
    latency = 8 # hours
    gfs_2_files = gfs_files_names[np.where(initial_dates_gfs_files - dt.timedelta(hours = 5) \
            <= forecast_initial_date - dt.timedelta(hours = latency))[0]][-2:]

    #Agregando path a estos archivos
    for idx in range(len(gfs_2_files)):
        gfs_2_files[idx] = path_data+'{0}/'.format(gfs_2_files[idx][-5:-3]) + gfs_2_files[idx]
    


    
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


def get_inputs_for_historic(station, forecast_initial_date):
    gfs_path = '/var/data1/AQ_Forecast_DATA/historic/GFS/historic/'
    cams_path = '/var/data1/AQ_Forecast_DATA/historic/CAMS/Pronostico/'
    path_IFRP ="/var/data1/AQ_Forecast_DATA/historic/GFS/historic/Vientos/BT/IFRP/"
    files_ifrp = np.sort(glob.glob(path_IFRP+'*'))
    dates_ifrp = np.array([dt.datetime.strptime(files_ifrp[i].split('/')[-1][:-4],'%Y%m%d%H')\
        for i in range(len(files_ifrp))])

    coor_esta= pd.read_csv("/var/data1/AQ_Forecast_DATA/historic/PM25/CoordenadasEstaciones.csv", 
                           index_col= "Nombre")
    lat_est = coor_esta.loc[station].Latitud
    lon_est = coor_esta.loc[station].Longitud

    # Get PM2.5

    pm2p5 = get_pm2p5_for_forecast(forecast_initial_date-dt.timedelta(hours=1),station_name=station)
    pm2p5.index = pm2p5.index+dt.timedelta(hours=1)

    ### Get GFS and CAMS ###
    df_GFS,df_optimal_gfs = get_gfs_for_historic(forecast_initial_date,\
        gfs_path,latlon = (lat_est,lon_est))
    df_CAMS,df_optimal_cams = get_cams_for_forecast(forecast_initial_date,\
        cams_path,operational = False,latlon = (lat_est,lon_est))

    ### Get IFRP
    file_ifrp = files_ifrp[dates_ifrp<=forecast_initial_date+\
               dt.timedelta(hours=5)-dt.timedelta(hours=5)][-1]
    df_IFRP = pd.read_csv(file_ifrp,index_col=1,parse_dates=True)[['IFRP']]
    df_IFRP.index = df_IFRP.index-dt.timedelta(hours=5)

    # Lagged data
    x_shifts = pm2p5.loc[forecast_initial_date-dt.timedelta(hours=47):forecast_initial_date].values[:,0]

    index_future = pd.date_range(forecast_initial_date+dt.timedelta(hours=1),\
                  forecast_initial_date+dt.timedelta(hours=96),freq='H')

    # CAMS
    cams_future = df_optimal_cams.loc[index_future[0]:].iloc[np.arange(0,97,3),:]
    aod_future = cams_future['aod'].values

    # GFS
    gfs_future = df_optimal_gfs.loc[index_future[0]:].iloc[np.arange(0,97,3),:]
    tcc_future = gfs_future['tcc'].values
    prate_future = gfs_future['prate'].values
    hpbl_future = gfs_future['hpbl'].values

    # IFRP
    ifrp50_future = df_IFRP.rolling(3,center=True).max().loc[index_future[0]:].iloc[np.arange(0,97,3),0].values

    # HOD/DOW
    df_hour = pd.DataFrame(index_future.hour, index = index_future)
    hod1_future = np.sin(2*np.pi*(df_hour/24)).iloc[np.arange(0,24,1),0].values
    hod2_future = np.cos(2*np.pi*(df_hour/24)).iloc[np.arange(0,24,1),0].values

    df_dow = pd.DataFrame(index_future.dayofweek, index = index_future)
    dow1_future = np.sin(2*np.pi*(df_dow/7)).iloc[np.arange(0,96,24),0].values
    dow2_future = np.cos(2*np.pi*(df_dow/7)).iloc[np.arange(0,96,24),0].values

    x_temp = np.hstack([x_shifts,
        aod_future,
        tcc_future,
        prate_future,
        hpbl_future,
        ifrp50_future,
        dow1_future,
        dow2_future,
        hod1_future,
        hod2_future])
    
    return index_future, x_temp, pm2p5, df_GFS, df_CAMS, df_IFRP


def get_inputs_for_historic_nofires(station, forecast_initial_date):
    gfs_path = '/var/data1/AQ_Forecast_DATA/historic/GFS/historic/'
    cams_path = '/var/data1/AQ_Forecast_DATA/historic/CAMS/Pronostico/'
    path_IFRP ="/var/data1/AQ_Forecast_DATA/historic/GFS/historic/Vientos/BT/IFRP/"
    files_ifrp = np.sort(glob.glob(path_IFRP+'*'))
    dates_ifrp = np.array([dt.datetime.strptime(files_ifrp[i].split('/')[-1][:-4],'%Y%m%d%H')\
        for i in range(len(files_ifrp))])

    coor_esta= pd.read_csv("/var/data1/AQ_Forecast_DATA/historic/PM25/CoordenadasEstaciones.csv", 
                           index_col= "Nombre")
    lat_est = coor_esta.loc[station].Latitud
    lon_est = coor_esta.loc[station].Longitud

    # Get PM2.5

    pm2p5 = get_pm2p5_for_forecast(forecast_initial_date-dt.timedelta(hours=1),station_name=station)
    pm2p5.index = pm2p5.index+dt.timedelta(hours=1)

    ### Get GFS and CAMS ###
    df_GFS,df_optimal_gfs = get_gfs_for_historic(forecast_initial_date,\
        gfs_path,latlon = (lat_est,lon_est))
    df_CAMS,df_optimal_cams = get_cams_for_forecast(forecast_initial_date,\
        cams_path,operational = False,latlon = (lat_est,lon_est))

    ### Get IFRP
    file_ifrp = files_ifrp[dates_ifrp<=forecast_initial_date+\
               dt.timedelta(hours=5)-dt.timedelta(hours=5)][-1]
    df_IFRP = pd.read_csv(file_ifrp,index_col=1,parse_dates=True)[['IFRP']]
    df_IFRP.IFRP = df_IFRP.IFRP * 0    #Haciendo cero por los incendios
    df_IFRP.index = df_IFRP.index-dt.timedelta(hours=5)

    # Lagged data
    x_shifts = pm2p5.loc[forecast_initial_date-dt.timedelta(hours=47):forecast_initial_date].values[:,0]

    index_future = pd.date_range(forecast_initial_date+dt.timedelta(hours=1),\
                  forecast_initial_date+dt.timedelta(hours=96),freq='H')

    # CAMS
    cams_future = df_optimal_cams.loc[index_future[0]:].iloc[np.arange(0,97,3),:]
    aod_future = cams_future['aod'].values

    # GFS
    gfs_future = df_optimal_gfs.loc[index_future[0]:].iloc[np.arange(0,97,3),:]
    tcc_future = gfs_future['tcc'].values
    prate_future = gfs_future['prate'].values
    hpbl_future = gfs_future['hpbl'].values

    # IFRP
    ifrp50_future = df_IFRP.rolling(3,center=True).max().loc[index_future[0]:].iloc[np.arange(0,97,3),0].values

    # HOD/DOW
    df_hour = pd.DataFrame(index_future.hour, index = index_future)
    hod1_future = np.sin(2*np.pi*(df_hour/24)).iloc[np.arange(0,24,1),0].values
    hod2_future = np.cos(2*np.pi*(df_hour/24)).iloc[np.arange(0,24,1),0].values

    df_dow = pd.DataFrame(index_future.dayofweek, index = index_future)
    dow1_future = np.sin(2*np.pi*(df_dow/7)).iloc[np.arange(0,96,24),0].values
    dow2_future = np.cos(2*np.pi*(df_dow/7)).iloc[np.arange(0,96,24),0].values

    x_temp = np.hstack([x_shifts,
        aod_future,
        tcc_future,
        prate_future,
        hpbl_future,
        ifrp50_future,
        dow1_future,
        dow2_future,
        hod1_future,
        hod2_future])
    
    return index_future, x_temp, pm2p5, df_GFS, df_CAMS, df_IFRP