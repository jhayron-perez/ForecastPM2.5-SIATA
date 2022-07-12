import xarray as xr
import numpy as np
from glob import glob
import pandas as pd
import datetime
import urllib.request
import sys, os
import requests
import netCDF4 as nc
import matplotlib.pyplot as plt
print('1')
sys.path.insert(1,'/home/jsperezc/jupyter/AQ_Forecast/functions/')
from data_downloading import procesa_gribs_GFS_a_netCDF_Wind
from data_downloading import procesa_gribs_GFS_a_netCDF
import requests
print('2')
def check_file_status(filepath, filesize):
    sys.stdout.write('\r')
    sys.stdout.flush()
    size = int(os.stat(filepath).st_size)
    percent_complete = (size/filesize)*100
    sys.stdout.write('%.3f %s' % (percent_complete, '% Completed'))
    sys.stdout.flush()

lon_min = -80
lon_max = -70
lat_min = -10
lat_max = 12

lon_min_w = -92
lon_max_w = -34
lat_min_w = -30
lat_max_w = 20

date = datetime.datetime.now() # %%time
hoy = date.strftime('%Y%m%d')
hora = datetime.datetime.today().hour

## Disponibilidad de las ejecuciones de GFS
## 00 UTC: Desde las 23 del día anterior hasta las 4am hora local
## 06 UTC: Desde las 5 hasta las 10am hora local
## 12 UTC: Desde las 11 hasta las 16 hora local
## 18 UTC: Desde las 17 hasta las 22 hora local

corridas = {5:"06", 6:"06",7:"06",8:"06",9:"06", 10:"06", 
            11:"12", 12:"12", 13:"12", 14:"12", 15:"12", 16:"12", 
            17:"18", 18:"18", 19:"18", 20:"18", 21:"18", 22: "18",
            23: "00", 0: "00", 1: "00", 2: "00", 3: "00", 4:"00"} ## Esto es para que dependiendo de la hora del día a las que se corra el código, él sepa cual corrida es la que debe descargar

if hora == 23:
    hoy = (date + datetime.timedelta(days = 1)).strftime('%Y%m%d')
print('3')
lista_descarga = ['https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{date}/{corrida}/atmos/gfs.t{corrida}z.pgrb2.0p25.f{h}'.format(date = hoy, corrida=corridas[hora], h = str(i).zfill(3)) for i in range(0, 121, 3)]

print(lista_descarga)

os.makedirs('/var/data1/AQ_Forecast_DATA/test/GFS/raw/%s/'%hoy, exist_ok = True)

for file in lista_descarga:
    filename = file
    file_base = os.path.basename(file)
    
    print('Downloading',file_base)
    urllib.request.urlretrieve(filename, '/var/data1/AQ_Forecast_DATA/test/GFS/raw/%s/'%hoy + file_base)


rutas_grib = sorted(glob('/var/data1/AQ_Forecast_DATA/test/GFS/raw/{}/*t{}z*'.format(hoy,corridas[hora])))
rutas_grib = [i for i in rutas_grib if "idx" not in i ]

print(rutas_grib)
try:
    ruta_save = '/var/data1/AQ_Forecast_DATA/historic/GFS/historic/{}/gfs_0p25_{}{}.nc'.format(corridas[hora], hoy, corridas[hora])
    print(ruta_save)
    procesa_gribs_GFS_a_netCDF(rutas_grib[1:], ruta_save)
    ruta_save_wind = '/var/data1/AQ_Forecast_DATA/historic/GFS/historic/Vientos/gfs_0p25_{}{}.nc'.format(hoy, corridas[hora])
    print(ruta_save_wind)
    procesa_gribs_GFS_a_netCDF_Wind(rutas_grib, ruta_save_wind)
    os.system("cp {} /var/data1/AQ_Forecast_DATA/operational/GFS/".format(ruta_save))
    os.system("cp {} /var/data1/AQ_Forecast_DATA/operational/GFS/Vientos/".format(ruta_save_wind))
    [os.remove(i) for i in glob('/var/data1/AQ_Forecast_DATA/test/GFS/raw/%s/*'%hoy)]
    os.rmdir('/var/data1/AQ_Forecast_DATA/test/GFS/raw/%s'%hoy)
    
    #remover archivos gfs de la carpeta operacional
    lista = sorted(glob('/var/data1/AQ_Forecast_DATA/operational/GFS/*gfs*'))
    if len(lista)>45:
        os.remove(lista[0])
        
    #remover vientos de la carpeta operacional
    lista_vientos = sorted(glob('/var/data1/AQ_Forecast_DATA/operational/GFS/Vientos/*gfs*'))
    dates_vientos_op = np.array([datetime.datetime.strptime(lista_vientos[i].split('/')[-1],'gfs_0p25_%Y%m%d%H.nc')\
         for i in range(len(lista_vientos))])
    vientos_remover = np.array(lista_vientos)[dates_vientos_op<date-datetime.timedelta(days=15)]
    for i in range(len(vientos_remover)):
        os.system('rm '+vientos_remover[i])
except Exception as e:
    print(hoy)
    print(e)
   
    