#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os, glob
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xrc
import datetime as dt
import gc
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point
import shapely.geometry as ss
import matplotlib.pyplot as plt

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import sys
sys.path.insert(1,'/home/jsperezc/jupyter/AQ_Forecast/functions/')
from trajectories import *

lati =   6.25
loni = -75.60

# Propiedades del cálculo
delta_t = 1   # horas
leveli  = 800 # hPa    

source = 'GFS'

path_files = '/var/data1/AQ_Forecast_DATA/historic/GFS/historic/Vientos/'
path_out   = '/var/data1/AQ_Forecast_DATA/historic/GFS/historic/Vientos/BT/'

ndays = 4

# Variables extraidas de los datos
files_date_2, files_list_2, lat_dataset, lon_dataset, levels_dataset = \
     Follow(path=path_files, path_fig=None, source=source,
     lati=lati,loni=loni, warnings=False)

#timerange = dt.datetime.today()
hoy = dt.datetime.today()
date_name = f'{hoy.strftime("%Y%m%d")}'
arch_hoy = np.sort(glob.glob(path_files + "*"+date_name+"*"))
ult_fecha = arch_hoy[-1].split("/")[-1][9:-3]
fecha_dt = dt.datetime.strptime(ult_fecha, "%Y%m%d%H")

for i, date_i in enumerate([fecha_dt]):
    print(date_i)
    # Nombre del archivo, correspondiente al día evaluado
    date_name = f'{date_i.strftime("%Y%m%d%H")}'
    arch_viento_actual = f"gfs_0p25_{date_name}.nc"
    aa = np.where(files_list_2<= path_files + arch_viento_actual)[0]
    files_list = files_list_2[aa]
    files_date = files_date_2[aa]
#        print("shapeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", files_date.shape)


    Fechai = date_i.strftime('%Y-%m-%d %H:00')
    Fechaf = (date_i+dt.timedelta(hours = 115)).strftime('%Y-%m-%d %H:00') ## Este 96 es por las 96 horas del pronóstico
#        print(f'{Fechai} -----> {Fechaf}')
    name_file = f'BT_GFS.{delta_t}h.{leveli}hPa.{date_name}.{ndays}days.nc'


    # Calculamos la retrotrayectoria para un solo día
    BT = Trajectories_level_i(lati=lati,loni=loni,leveli=leveli,Fechai=Fechai, Fechaf=Fechaf, delta_t=1,
        ndays=ndays, source='GFS',path_files=path_files,
        files_date=files_date, files_list=files_list, 
        lat_dataset=lat_dataset, lon_dataset=lon_dataset, 
        levels_dataset=levels_dataset)

    # Se almacena el nuevo archivo  NetCDF4 
    save_nc(dictionary=BT, file_out=f'{path_out}{name_file}')
    del(BT)
    gc.collect()
    
df_fires = pd.read_csv("/var/data1/AQ_Forecast_DATA/operational/Fires/MODIS_C6_1_South_America_7d.csv")
######### Incendios para los últimos 7 días ############
str_time = df_fires['acq_time'].values.astype(str)
str_time = np.array([str_time[i].zfill(4) for i in range(len(str_time))])
str_date = df_fires['acq_date'].values.astype(str)

dates_fires = np.array([dt.datetime.strptime(str_date[i]+' '+str_time[i],'%Y-%m-%d %H%M') \
    for i in range(len(str_time))])
df_fires.index = dates_fires
df_fires.index = df_fires.index# - dt.timedelta(hours = 5)
############################################################

fechas, plev_values, lat_values, lon_values = read_nc(f'{path_out}{name_file}')
#fechas = fechas[~np.isnan(fechas)]
[dates_dim, back_step_dim] = fechas.shape
#print("arch leido")
'''
dates_dim:      dimensión de las fechas a partir de las cuales se van a 
                calcular las trayectorias
back_step_dim:  dimensión de las fechas en retroceso de la retrotrayectoria 
''' 
days_b = 4
m_buffer = 500_000
list_IFRP = []
for dt_i in (range(dates_dim)):
    fila_IFRP=[]
    fila_IFRP.append(str(fechas[:,0][dt_i]))
    print(str(fechas[:,0][dt_i]))
    ind_dias_atras = days_b*24 ## 4 porque fueron los días hacia atrás seleccionados finalmente 
    lat_i = lat_values[dt_i,:ind_dias_atras] #todas las latitudes de la retrotrayectoria iniciada en el timepo dt_i
    lon_i = lon_values[dt_i,:ind_dias_atras]        
    lat_i = lat_i[~np.isnan(lat_i)]
    lon_i = lon_i[~np.isnan(lon_i)]
    #geom_list = [(x, y) for x, y  in zip([lon_i], [lat_i])]
    geom_list_2 = LineString((zip(lon_i, lat_i)))
    
    fechas_i = fechas[dt_i][~np.isnan(fechas[dt_i])]
    fechas_i = fechas_i[:ind_dias_atras]
    grado = meters_to_degrees(m_buffer) ## Los kilómetros del buffer
    poligon_buffer = geom_list_2.buffer(grado)
    poligon_buffer
    IFRP = 0
    
    fecha_ini = pd.to_datetime(fechas[0][0]) - dt.timedelta(days=4.5)
    fires_retro = df_fires[str(fecha_ini):str(fechas[0][0])] 
    for fire in range(len(fires_retro)):
        lat_fire = fires_retro["latitude"][fire]
        lon_fire = fires_retro["longitude"][fire]
        aa = search_fire(lon_fire, lat_fire, poligon_buffer)
        if aa:
            IFRP = IFRP+ fires_retro["frp"][fire]
    fila_IFRP.append(IFRP)
    list_IFRP.append(fila_IFRP)
    
path_out_IFRP ="/var/data1/AQ_Forecast_DATA/historic/GFS/historic/Vientos/BT/IFRP/"
df_IFRP = pd.DataFrame(list_IFRP, columns = ["fecha", "IFRP"])
df_IFRP.to_csv(path_out_IFRP + f"{date_name}.csv")