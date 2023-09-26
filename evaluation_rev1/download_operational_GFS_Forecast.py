# %%
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
#from data_downloading import procesa_gribs_GFS_a_netCDF_Wind
#from data_downloading import procesa_gribs_GFS_a_netCDF
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


### ---- Funciones para extraer la información (versión modificada) ----

#Para pasar gribs de GFS a nc
def procesa_gribs_GFS_a_netCDF(rutas_grib, ruta_salida):
    import xarray as xr
    import pandas as pd
    import numpy as np
    import os
    
    lon_min = -80
    lon_max = -70
    lat_min = -10
    lat_max = 12

    lon_min_w = -92
    lon_max_w = -34
    lat_min_w = -30
    lat_max_w = 20
    
    tcc_un_avg_l, tcc_un_ins_l, tcc_atm_avg_l, rad_in_l, tp_srf_acum_l, prate_srf_avg_l, prate_srf_ins_l, hpbl_l, cin_l, vv_np_l, uu_np_l, fechas = [],[], [], [],[], [],[],[], [], [], [], []
    for ruta_grib in rutas_grib[:]:
        print(ruta_grib)
        
        tcc_un_avg = xr.open_dataset(ruta_grib, engine = 'cfgrib', backend_kwargs={'filter_by_keys':{'typeOfLevel': 'unknown', 'stepType': 'avg', 'shortName': 'tcc'}})
        tcc_un_ins = xr.open_dataset(ruta_grib, engine = 'cfgrib', backend_kwargs={'filter_by_keys':{'typeOfLevel': 'unknown', 'stepType': 'avg', 'shortName': 'tcc'}})
        tcc_atm_avg = xr.open_dataset(ruta_grib, engine = 'cfgrib', backend_kwargs={'filter_by_keys':{'typeOfLevel': 'atmosphere', 'stepType': 'avg', 'shortName': 'tcc'}})
#         print("antes de rada")
        rad_in = xr.open_dataset(ruta_grib, engine = 'cfgrib', backend_kwargs={'filter_by_keys':{'typeOfLevel': 'surface', "shortName": "dswrf"}})
#         print("despues de radia")
        tp_srf_acum = xr.open_dataset(ruta_grib, engine = 'cfgrib', backend_kwargs={'filter_by_keys':{'typeOfLevel': 'surface', 'stepType': 'accum', 'shortName': 'tp'}})
        prate_srf_avg = xr.open_dataset(ruta_grib, engine = 'cfgrib', backend_kwargs={'filter_by_keys':{'typeOfLevel': 'surface', 'stepType': 'avg',"shortName":"prate"}})
        hpbl = xr.open_dataset(ruta_grib, engine = 'cfgrib', backend_kwargs={'filter_by_keys':{'typeOfLevel': 'surface',"shortName":"hpbl"}})
        cin = xr.open_dataset(ruta_grib, engine = 'cfgrib', backend_kwargs={'filter_by_keys':{'typeOfLevel': 'surface', 'shortName': 'cin'}})
        
        ##Variables con datos en la vertical --> 'typeOfLevel': 'isobaricInhPa'
        vv_np = xr.open_dataset(ruta_grib, engine = 'cfgrib', backend_kwargs={'filter_by_keys':{'typeOfLevel': 'isobaricInhPa', 'shortName': 'v'}})
        uu_np = xr.open_dataset(ruta_grib, engine = 'cfgrib', backend_kwargs={'filter_by_keys':{'typeOfLevel': 'isobaricInhPa', 'shortName': 'u'}})
#         print("se leyeron todas las variables")
        longitudes = tcc_un_avg.longitude.values
        latitudes = tcc_un_avg.latitude.values
#         print("la cargué")
        #print(latitudes)
        longitudes[longitudes > 180] = longitudes[longitudes > 180] - 360

        mascara_latitudes = (latitudes >= lat_min)&(latitudes <= lat_max)
        mascara_longitudes = (longitudes >= lon_min)&(longitudes <= lon_max)
        ## Para udar el nivel de presión de interés 
        niveles =[900]
        mascara_niveles = (vv_np.isobaricInhPa.values == niveles)
        #print("wwwwwwwwwwwwwwwwwwwww", w)
        tcc_un_avg_l.append(tcc_un_avg.tcc[mascara_latitudes, :].values[:, mascara_longitudes])
        tcc_un_ins_l.append(tcc_un_ins.tcc[mascara_latitudes, :].values[:, mascara_longitudes])
        tcc_atm_avg_l.append(tcc_atm_avg.tcc[mascara_latitudes, :].values[:, mascara_longitudes])
        rad_in_l.append(rad_in.dswrf[mascara_latitudes, :].values[:, mascara_longitudes])
        tp_srf_acum_l.append(tp_srf_acum.tp[mascara_latitudes, :].values[:, mascara_longitudes])
        prate_srf_avg_l.append(prate_srf_avg.prate[mascara_latitudes, :].values[:, mascara_longitudes])
        hpbl_l.append(hpbl.hpbl[mascara_latitudes, :].values[:, mascara_longitudes])
        cin_l.append(cin.cin[mascara_latitudes, :].values[:, mascara_longitudes])
        print(np.array(hpbl_l).shape)
        
        ## Las variables en la vertical están en 3 dimensiones 
        aa_v = vv_np.v[mascara_niveles ,mascara_latitudes, mascara_longitudes].values
        vv_np_l.append(aa_v)
        aa_u = uu_np.u[mascara_niveles ,mascara_latitudes, mascara_longitudes].values
        uu_np_l.append(aa_u)
        print("New date:")
        print(ruta_grib)
        ### --- Old V
        # date = pd.to_datetime(os.path.dirname(ruta_grib).split('/')[-1], format = '%Y%m%d') + \
        #     pd.Timedelta(os.path.basename(ruta_grib).split('t')[1][:2] + 'H')+\
        #     pd.Timedelta(os.path.basename(ruta_grib).split('f')[-1] + 'H')
        ### --- New V
        date = pd.to_datetime(os.path.dirname(ruta_grib).split('/')[-1], format = '%Y%m%d') + \
            pd.Timedelta(os.path.basename(ruta_grib).split('.')[2][-2:] + 'H')+\
            pd.Timedelta(os.path.basename(ruta_grib).split('.')[3][1:] + 'H')

        print(date)
        fechas.append(date)
        tcc_un_avg.close()
        tcc_un_ins.close()
        tcc_atm_avg.close()
        rad_in.close()
        tp_srf_acum.close()
        prate_srf_avg.close()
        hpbl.close()
        cin.close()
        uu_np.close()
        vv_np.close()
#         print("cerré todo")
    data = {}
    data['tcc_un_avg'] = xr.DataArray(tcc_un_avg_l,
                 coords = [fechas,
                           latitudes[mascara_latitudes],
                           longitudes[mascara_longitudes]],
                 dims = ['time', 'latitude', 'longitude'],
                 attrs = {i:tcc_un_avg.tcc.attrs[i] for i in ['long_name', 'units']})
    
    data['tcc_un_ins'] = xr.DataArray(tcc_un_ins_l,
                 coords = [fechas,
                           latitudes[mascara_latitudes],
                           longitudes[mascara_longitudes]],
                 dims = ['time', 'latitude', 'longitude'],
                 attrs = {i:tcc_un_avg.tcc.attrs[i] for i in ['long_name', 'units']})
    
    data['tcc_atm_avg'] = xr.DataArray(tcc_atm_avg_l,
                 coords = [fechas,
                           latitudes[mascara_latitudes],
                           longitudes[mascara_longitudes]],
                 dims = ['time', 'latitude', 'longitude'],
                 attrs = {i:tcc_un_avg.tcc.attrs[i] for i in ['long_name', 'units']})
    
    
    data['rad_in'] = xr.DataArray(rad_in_l,
                 coords = [fechas,
                           latitudes[mascara_latitudes],
                           longitudes[mascara_longitudes]],
                 dims = ['time', 'latitude', 'longitude'],
                 attrs = {i:rad_in.dswrf.attrs[i] for i in ['long_name', 'units']})
    
    data['tp_srf_acum'] = xr.DataArray(tp_srf_acum_l,
                 coords = [fechas,
                           latitudes[mascara_latitudes],
                           longitudes[mascara_longitudes]],
                 dims = ['time', 'latitude', 'longitude'],
                 attrs = {i:tp_srf_acum.tp.attrs[i] for i in ['long_name', 'units']})
    
    data['prate_srf_avg'] = xr.DataArray(prate_srf_avg_l,
                 coords = [fechas,
                           latitudes[mascara_latitudes],
                           longitudes[mascara_longitudes]],
                 dims = ['time', 'latitude', 'longitude'],
                 attrs = {i:prate_srf_avg.prate.attrs[i] for i in ['long_name', 'units']})
    
    data['hpbl'] = xr.DataArray(hpbl_l,
                 coords = [fechas,
                           latitudes[mascara_latitudes],
                           longitudes[mascara_longitudes]],
                 dims = ['time', 'latitude', 'longitude'],
                 attrs = {i:hpbl.hpbl.attrs[i] for i in ['long_name', 'units']})

    data['cin'] = xr.DataArray(cin_l,
                 coords = [fechas,
                           latitudes[mascara_latitudes],
                           longitudes[mascara_longitudes]],
                 dims = ['time', 'latitude', 'longitude'],
                 attrs = {i:cin.cin.attrs[i] for i in ['long_name', 'units']})
    
    vv_np_l = (np.array(vv_np_l).reshape(np.array(vv_np_l).shape[0], np.array(vv_np_l).shape[2], np.array(vv_np_l).shape[3])).tolist()
    data['v_900'] = xr.DataArray(vv_np_l,
                 coords = [fechas,
                           latitudes[mascara_latitudes],
                           longitudes[mascara_longitudes]],
                 dims = ['time', 'latitude', 'longitude'],
                 attrs = {i:vv_np.v.attrs[i] for i in ['long_name', 'units']})
    uu_np_l = (np.array(uu_np_l).reshape(np.array(uu_np_l).shape[0], np.array(uu_np_l).shape[2], np.array(uu_np_l).shape[3])).tolist()
    data['u_900'] = xr.DataArray(uu_np_l,
                 coords = [fechas,
                           latitudes[mascara_latitudes],
                           longitudes[mascara_longitudes]],
                 dims = ['time', 'latitude', 'longitude'],
                 attrs = {i:uu_np.u.attrs[i] for i in ['long_name', 'units']})
    
    #print("soy el shape de data wwwwwwwwwwww",data["tcc_un_ins"].shape)
    ds = xr.Dataset(data, attrs = {'Conventions':'CF-1.7', 'institution':'US National Weather Service - NCEP'})


    encoding = {variable:{'zlib': True,'complevel': 9, 'dtype':'int8'} for variable in list(ds.data_vars.keys())}

    ds.to_netcdf(ruta_salida)#, encoding = encoding)


#Para pasar gribs de GFS a netCDF (vientos)
def procesa_gribs_GFS_a_netCDF_Wind(rutas_grib, ruta_salida):
    import xarray as xr
    import pandas as pd
    import numpy as np
    import os
    
    lon_min = -80
    lon_max = -70
    lat_min = -10
    lat_max = 12

    lon_min_w = -92
    lon_max_w = -34
    lat_min_w = -30
    lat_max_w = 20
    
    ws, us, vs, fechas = [], [], [], []

    for ruta_grib in rutas_grib[:]:

        w = xr.open_dataset(ruta_grib, engine = 'cfgrib', backend_kwargs={'filter_by_keys':{'typeOfLevel': 'isobaricInhPa', 'shortName': 'w'}})['w']
        u = xr.open_dataset(ruta_grib, engine = 'cfgrib', backend_kwargs={'filter_by_keys':{'typeOfLevel': 'isobaricInhPa', 'shortName': 'u'}})['u']
        v = xr.open_dataset(ruta_grib, engine = 'cfgrib', backend_kwargs={'filter_by_keys':{'typeOfLevel': 'isobaricInhPa', 'shortName': 'v'}})['v']

        longitudes = u.longitude.values
        latitudes = u.latitude.values
        longitudes[longitudes > 180] = longitudes[longitudes > 180] - 360

        niveles_presion = [1000,  975,  950,  925,  900,  850,  800,  750,  700,  650,  600,  550,
        500,  450,  400,  350,  300,  250,  200,  150,  100]#w.isobaricInhPa.values
        mascara_nivelesu = (u.isobaricInhPa.values >= 100)&(u.isobaricInhPa.values <= 1000)
        mascara_nivelesw = (w.isobaricInhPa.values >= 100)&(w.isobaricInhPa.values <= 1000)
        mascara_latitudes = (latitudes >= lat_min_w)&(latitudes <= lat_max_w)
        mascara_longitudes = (longitudes >= lon_min_w)&(longitudes <= lon_max_w)
        #print("wwwwwwwwwwwwwwwwwwwww", w)
        ws.append(w[mascara_nivelesw, :, :][:, :, mascara_longitudes].values[:, mascara_latitudes, :])
        us.append(u[mascara_nivelesu, :, :][:, :, mascara_longitudes].values[:, mascara_latitudes, :])
        vs.append(v[mascara_nivelesu, :, :][:, :, mascara_longitudes].values[:, mascara_latitudes, :])

        #date = pd.to_datetime(ruta_grib.split('.')[2], format = '%Y%m%d%H') + pd.Timedelta(hours = int(ruta_grib.split('.')[3][1:]))
        ### --- Old V
        # date = pd.to_datetime(os.path.dirname(ruta_grib).split('/')[-1], format = '%Y%m%d') + \
        #     pd.Timedelta(os.path.basename(ruta_grib).split('t')[1][:2] + 'H')+\
        #     pd.Timedelta(os.path.basename(ruta_grib).split('f')[-1] + 'H')
        ### --- New V
        date = pd.to_datetime(os.path.dirname(ruta_grib).split('/')[-1], format = '%Y%m%d') + \
            pd.Timedelta(os.path.basename(ruta_grib).split('.')[2][-2:] + 'H')+\
            pd.Timedelta(os.path.basename(ruta_grib).split('.')[3][1:] + 'H')
        
        print("ws size", len(ws))
        fechas.append(date)
        print(date)
        w.close()
        u.close()
        v.close()

    ws = np.array(ws)
#     print("soy wsssssssss:")
    print("ws array shape",ws.shape)
    us = np.array(us)
    vs = np.array(vs)

    data = {}
    
    data['w'] = xr.DataArray(ws,
                 coords = [fechas,
                           niveles_presion,
                           latitudes[mascara_latitudes],
                           longitudes[mascara_longitudes]],
                 dims = ['time', 'isobaricInhPa', 'latitude', 'longitude'],
                 attrs = {i:w.attrs[i] for i in ['long_name', 'units', 'standard_name']})

    data['u'] = xr.DataArray(us,
                 coords = [fechas,
                           niveles_presion,
                           latitudes[mascara_latitudes],
                           longitudes[mascara_longitudes]],
                 dims = ['time', 'isobaricInhPa', 'latitude', 'longitude'],
                 attrs = {i:u.attrs[i] for i in ['long_name', 'units', 'standard_name']})

    data['v'] = xr.DataArray(vs,
                 coords = [fechas,
                           niveles_presion,
                           latitudes[mascara_latitudes],
                           longitudes[mascara_longitudes]],
                 dims = ['time', 'isobaricInhPa', 'latitude', 'longitude'],
                 attrs = {i:v.attrs[i] for i in ['long_name', 'units', 'standard_name']})
    print("Data shape:",data["w"].shape)
    ds = xr.Dataset(data, attrs = {'Conventions':'CF-1.7', 'institution':'US National Weather Service - NCEP'})
    ds.to_netcdf(ruta_salida)#, encoding = encoding)

# date = datetime.datetime.now() # %%time
# hoy = date.strftime('%Y%m%d')
# hora = datetime.datetime.today().hour



## Disponibilidad de las ejecuciones de GFS
## 00 UTC: Desde las 23 del día anterior hasta las 4am hora local
## 06 UTC: Desde las 5 hasta las 10am hora local
## 12 UTC: Desde las 11 hasta las 16 hora local
## 18 UTC: Desde las 17 hasta las 22 hora local

# corridas = {5:"06", 6:"06",7:"06",8:"06",9:"06", 10:"06", 
#             11:"12", 12:"12", 13:"12", 14:"12", 15:"12", 16:"12", 
#             17:"18", 18:"18", 19:"18", 20:"18", 21:"18", 22: "18",
#             23: "00", 0: "00", 1: "00", 2: "00", 3: "00", 4:"00"} ## Esto es para que dependiendo de la hora del día a las que se corra el código, él sepa cual corrida es la que debe descargar

# if hora == 23:
#     hoy = (date + datetime.timedelta(days = 1)).strftime('%Y%m%d')
# print('3')

#Fechas de la evaluación
#Dates = pd.date_range('2020-03-01', '2020-06-01', freq='6H')
Dates = pd.date_range('2020-05-24 00:00', '2020-06-01 00:00', freq='6H')

#Recorriendo fechas
for pd_date in Dates:

    #Fecha a descargar
    date = pd_date.to_pydatetime()
    hoy = date.strftime('%Y%m%d')
    hora = str(date.hour).zfill(2)


    lista_descarga = ["https://data.rda.ucar.edu/ds084.1/{0}/{1}/gfs.0p25.{1}{2}.f{3}.grib2".format(date.year, hoy, hora, str(init).zfill(3)) for init in range(0, 121, 3)]
    # lista_descarga = ['https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{date}/{corrida}/atmos/gfs.t{corrida}z.pgrb2.0p25.f{h}'.format(
    #     date = hoy, corrida=corridas[hora], h = str(i).zfill(3)) for i in range(0, 121, 3)]

    print(lista_descarga)


    os.makedirs('/var/data1/AQ_Forecast_DATA/test/GFS/raw/%s/'%hoy, exist_ok = True)

    for file in lista_descarga:
        filename = file
        file_base = os.path.basename(file)
        
        print('Downloading',file_base)
        urllib.request.urlretrieve(filename, '/var/data1/AQ_Forecast_DATA/test/GFS/raw/%s/'%hoy + file_base)


    rutas_grib = sorted(glob('/var/data1/AQ_Forecast_DATA/test/GFS/raw/{0}/*{0}{1}*'.format(hoy,hora)))
    rutas_grib = [i for i in rutas_grib if "idx" not in i ]


    print(rutas_grib)
    try:
        ruta_save = '/var/data1/AQ_Forecast_DATA/historic/GFS/historic/{}/gfs_0p25_{}{}.nc'.format(hora, hoy, hora)
        print(ruta_save)
        procesa_gribs_GFS_a_netCDF(rutas_grib[1:], ruta_save)
        ruta_save_wind = '/var/data1/AQ_Forecast_DATA/historic/GFS/historic/Vientos/gfs_0p25_{}{}.nc'.format(hoy, hora)
        print(ruta_save_wind)
        procesa_gribs_GFS_a_netCDF_Wind(rutas_grib, ruta_save_wind)
        #os.system("cp {} /var/data1/AQ_Forecast_DATA/operational/GFS/".format(ruta_save))
        #os.system("cp {} /var/data1/AQ_Forecast_DATA/operational/GFS/Vientos/".format(ruta_save_wind))
        for i in glob('/var/data1/AQ_Forecast_DATA/test/GFS/raw/%s/*'%hoy):
            os.remove(i)
        #[os.remove(i) for i in glob('/var/data1/AQ_Forecast_DATA/test/GFS/raw/%s/*'%hoy)]
        os.rmdir('/var/data1/AQ_Forecast_DATA/test/GFS/raw/%s'%hoy)
        
        # #remover archivos gfs de la carpeta operacional
        # lista = sorted(glob('/var/data1/AQ_Forecast_DATA/operational/GFS/*gfs*'))
        # if len(lista)>45:
        #     os.remove(lista[0])
            
        #remover vientos de la carpeta operacional
        # lista_vientos = sorted(glob('/var/data1/AQ_Forecast_DATA/operational/GFS/Vientos/*gfs*'))
        # dates_vientos_op = np.array([datetime.datetime.strptime(lista_vientos[i].split('/')[-1],'gfs_0p25_%Y%m%d%H.nc')\
        #      for i in range(len(lista_vientos))])
        # vientos_remover = np.array(lista_vientos)[dates_vientos_op<date-datetime.timedelta(days=15)]
        # for i in range(len(vientos_remover)):
        #     os.system('rm '+vientos_remover[i])
    except Exception as e:
        print(hoy)
        print(e)
    

    print("Pronostico {0}{1}".format(hoy, hora), 'terminado!')
# %%
