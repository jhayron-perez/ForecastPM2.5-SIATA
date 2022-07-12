##Funcion para descargar los datos de GEOS5
def download_geos5(date_first, date_end, lon_lat, path_save):
    """
    Esta función permite descargar datos de GEOS5, específicamente datos de aerosoles.
    Debe ingresar la fecha inicial, la fecha final y la ruta de almacenamiento. Tenga en cuenta que se descargaran todos 
    los archivos horarios de un día. Además, si desea descargar periodos que "corten" los años, por ejemplo: 
    2016-11-01 a 2017-01-31, deberá descargar primero la parte del primer año y luego la otra parte del otro. 
    
    Los datos de GEOS5 vienen en formato de cada 3 horas, siendo la 1:30 hora inicial y las 22:30 la hora final. Asimismo, estos 
    datos vienen en una resolución espacial XXX para todo el globo. 
    
    Esta función permite, además, generar un "merge" entre los datos horarios, teniendo un archivo por día.
    
    Por otro lado, esta funcion genera un recorte de los datos en una región de interés, para esto debe ingresar un vector con       las coordenadas en las siguientes posiciones: [N,S,E,O].
    
    Al final, se eliminan los archivos horarios originales de la descarga.
    """
    import wget
    import numpy as np
    import os
    import glob
    import pandas as pd
    import xarray as xr
    years = np.arange(pd.Period(date_first).year,pd.Period(date_end).year+1)
    months = np.arange(pd.Period(date_first).month,pd.Period(date_end).month+1)    
    days = np.arange(pd.Period(date_first).day,pd.Period(date_end).day+1)
    hours = np.arange(1,23,3)
    
    #Creamos un directorio donde se almacenaran los datos 
    os.system('mkdir {}GEOS5'.format(path_save))
    
    path_save = path_save+'GEOS5/'
    
    #loop for years
    for year in years:
        os.system('mkdir {}{}'.format(path_save,year))
        #loop for months
        for j in months:
            if j < 10:
                month = '0'+str(j)
            else:
                month = str(j)
            os.system('mkdir {}{}/{}'.format(path_save,year,month))
            #loop for days:
            for k in days:
                if k < 10:
                    day = '0'+str(k)
                else:
                    day = str(k)
                #os.system('mkdir {}{}/{}/{}'.format(path_save,year,month,day))

                #Indicamos la direccion donde se almacenaran los archivos
                path_files = '{}{}/{}/'.format(path_save,year,month)
                #loop for hours:
                for h in hours:
                    if h < 10:
                        hour = '0'+str(h)
                    else:
                        hour = str(h)
                    try:
                        url='https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/Y'+str(year)+'/M'+str(month)+'/D'+str(day)+'/GEOS.fp.asm.tavg3_2d_aer_Nx.'+str(year)+str(month)+str(day)+'_'+hour+'30.V01.nc4' #url for download file
                        print(url)
                        filename = wget.download(url, out= path_files)
                    except:
                        print('No se descargó el archivo {}-{}-{} {}:30'.format(year,month,day,hour))
                try:
                    #Hacemos un merge con todos los archivos horarios para tener archivos diarios
                    files = glob.glob(path_files+'/GEOS.fp.asm.tavg3_2d_aer_Nx*') #Creamos un vector con la ruta de cada
                    #archivo decargado
                    data=xr.open_mfdataset(files, concat_dim='time', autoclose='True') #Hacemos el merge con xarray

                    lon = data.coords["lon"]
                    lat = data.coords["lat"]

                    #Hacemos el recorte de los datos a nuestra region de interés
                    data = data.loc[dict(lon=lon[(lon > lon_lat[3]) & (lon < lon_lat[2])], 
                                         lat=lat[(lat > lon_lat[1]) & (lat < lon_lat[0])])] 

                    data.to_netcdf(path_files+'GEOS5_AOD_{}-{}-{}.nc'.format(year,month,day), 
                                   unlimited_dims='time',format="NETCDF4_CLASSIC") #Guardamos el nuevo archivos como NetCDF

                    print('='*70)
                    print('Eliminamos los archivos horarios')
                    os.system('rm {}GEOS.fp.asm.tavg3_2d_aer_Nx*'.format(path_files))
                except:
                    print('No existe o no se logró abrir y unir los archivos {}-{}-{}'.format(year,month,day))
    os.system('rm *.tmp')
    
    
# Funcion para descargar los datos de CAMS tanto de reanalisis como de pronostico    
def download_cams(date_first, date_end, lon_lat, path_save, datos):
    import numpy
    import cdsapi
    c = cdsapi.Client()
    import numpy as no
    import pandas as pd
    import datetime
    import os
    from dateutil.relativedelta import relativedelta
    
    #creamos la careta donde se almacenaran los datos de CAMS
    os.system('mkdir {}CAMS') 
    
    if datos == 'Pronostico':
        #Creamos un vector de fechas con rango mensual
        fecs = pd.date_range(Fechai, Fechaf, freq = 'M')

        for date in fecs:        
            #Descargamos los archivos por mes
            os.system('mkdir {}CAMS/{}/'.format(path_save,datos))
            #Se  crea la carpeta del año
            os.system('mkdir {}CAMS/{}/{}/'.format(path_save,datos,date.year))

            #Contruimos el rango de fechas para descargar cada mes
            fec_fin = date
            if date.day == 31:      
                fec_ini = (date - relativedelta(months=1))+datetime.timedelta(days=1)
            elif date.day == 30:
                fec_ini = (date - relativedelta(months=1))+datetime.timedelta(days=2)
            elif date.day == 29:
                fec_ini = (date - relativedelta(months=1))+datetime.timedelta(days=3)
            elif date.day == 28:
                fec_ini = (date - relativedelta(months=1))+datetime.timedelta(days=4)
            print('fecha inicial {}'.format(fec_ini))
            print('fecha final {}'.format(fec_fin))
            print('-'*20)

            if date.month < 10:
                month = '0'+str(date.month)
            else:
                month = str(date.month)
            os.system('mkdir {}CAMS/{}/{}/{}'.format(path_save,datos,date.year,month))


            c.retrieve(
                'cams-global-atmospheric-composition-forecasts',
                {
                    'type': 'forecast',
                    'format': 'netcdf_zip',
                    'variable': [
                        'black_carbon_aerosol_optical_depth_550nm', 'dust_aerosol_optical_depth_550nm', 
                        'organic_matter_aerosol_optical_depth_550nm',
                        'particulate_matter_10um', 'particulate_matter_2.5um', 'sea_salt_aerosol_optical_depth_550nm',
                        'total_aerosol_optical_depth_550nm',
                    ],
                    'date': '{}/{}'.format(fec_ini.strftime("%Y-%m-%d"),fec_fin.strftime("%Y-%m-%d")),
                    'time': '00:00',
                    'leadtime_hour': [
                        '0', '102', '108',
                        '114', '12', '120',
                        '18', '24', '30',
                        '36', '42', '48',
                        '54', '6', '60',
                        '66', '72', '78',
                        '84', '90', '96',
                        '102', '108', '114',
                        '120',
                    ],
                    'area': lon_lat,
                },
                '{}CAMS/{}/{}/{}/CAMS_{}.netcdf_zip'.format(path_save,datos,date.year,month,date.date()))

            # Descomprimimos el archivo zip
            #os.system('{}CAMS/{}/{}/{}/CAMS_{}.netcdf_zip'.format(path_save,datos,date.year,month,date.date()))
            os.system('unzip -o -q {}CAMS/{}/{}/{}/*zip'.format(path_save,datos,date.year,month))
            # Eliminamos el archivo zip
            #os.system('rm {}CAMS/{}/{}/{}/CAMS_{}.netcdf_zip'.format(path_save,datos,date.year,month,date.date()))
                #'/var/data1/AQ_Forecast_DATA/test/CAMS/test2016_feb-mar.netcdf_zip'
                #'/var/data1/AQ_Forecast_DATA/historic/CAMS/data_2016_01.netcdf_zip')
                
    elif datos == 'Reanalisis':
        #Creamos un vector de fechas con rango diario
        fecs = pd.date_range(Fechai, Fechaf, freq = 'D')
        
        for date in fecs:        
            #Descargamos los archivos por dia
            os.system('mkdir {}CAMS/{}/'.format(path_save,datos))
            #Se  crea la carpeta del año
            os.system('mkdir {}CAMS/{}/{}/'.format(path_save,datos,date.year))
            
            if date.month < 10:
                month = '0'+str(date.month)
            else:
                month = str(date.month)
            os.system('mkdir {}CAMS/{}/{}/{}'.format(path_save,datos,date.year,month))
            
            c.retrieve(
                'cams-global-reanalysis-eac4',
                {
                    'format': 'netcdf',
                    'variable': [
                        'black_carbon_aerosol_optical_depth_550nm', 'dust_aerosol_optical_depth_550nm', 
                        'organic_matter_aerosol_optical_depth_550nm',
                        'particulate_matter_10um', 'particulate_matter_2.5um', 'sea_salt_aerosol_optical_depth_550nm',
                        'total_aerosol_optical_depth_550nm',
                    ],
                    'date': '{}/{}'.format(date.date(),date.date()),
                    'time': [
                            '00:00', '03:00', '06:00',
                            '09:00', '12:00', '15:00',
                            '18:00', '21:00',
                        ],
                    'area': lon_lat,
                },
                '{}CAMS/{}/{}/{}/CAMS_{}.nc'.format(path_save,datos,date.year,month,date.date()))
            
def download_cams_forecast(datetime_forecast, lon_lat, path_save,datos = 'Pronostico'):
    import numpy
    import cdsapi
    c = cdsapi.Client()
    import numpy as np
    import glob
    import pandas as pd
    import datetime
    import os
    from dateutil.relativedelta import relativedelta
    
    #creamos la careta donde se almacenaran los datos de CAMS
    os.system(f'mkdir {path_save}CAMS') 
    str_date = str(datetime_forecast)
    print('{}'.format(str(datetime_forecast.date())))
    print('{}'.format(str(str_date)[11:-3]))
    #Descargamos los archivos por mes
    os.system('mkdir {}CAMS/{}/'.format(path_save,datos))
    #Se  crea la carpeta del año
    os.system('mkdir {}CAMS/{}/{}/'.format(path_save,datos,datetime_forecast.year))

    #Contruimos el rango de fechas para descargar cada mes
    month = str(datetime_forecast.month).zfill(2)
    os.system('mkdir {}CAMS/{}/{}/{}'.format(path_save,datos,datetime_forecast.year,month))
    print('mkdir {}CAMS/{}/{}/{}'.format(path_save,datos,datetime_forecast.year,month))
    c.retrieve(
        'cams-global-atmospheric-composition-forecasts',
        {
            'type': 'forecast',
            'format': 'netcdf_zip',
            'variable': [
                'black_carbon_aerosol_optical_depth_550nm', 'dust_aerosol_optical_depth_550nm', 
                'organic_matter_aerosol_optical_depth_550nm',
                'particulate_matter_10um', 'particulate_matter_2.5um', 'sea_salt_aerosol_optical_depth_550nm',
                'total_aerosol_optical_depth_550nm', 
                'nitrate_aerosol_optical_depth_550nm','ammonium_aerosol_optical_depth_550nm',
                'sulphate_aerosol_optical_depth_550nm',
            ],
            'date': '{}'.format(datetime_forecast.date()),
            'time': '{}'.format(str(str_date)[11:-3]),
            'leadtime_hour': [
                '0', '1', '10',
                '100', '101', '102',
                '103', '104', '105',
                '106', '107', '108',
                '109', '11', '110',
                '111', '112', '113',
                '114', '115', '116',
                '117', '118', '119',
                '12', '120', '13',
                '14', '15', '16',
                '17', '18', '19',
                '2', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '3',
                '30', '31', '32',
                '33', '34', '35',
                '36', '37', '38',
                '39', '4', '40',
                '41', '42', '43',
                '44', '45', '46',
                '47', '48', '49',
                '5', '50', '51',
                '52', '53', '54',
                '55', '56', '57',
                '58', '59', '6',
                '60', '61', '62',
                '63', '64', '65',
                '66', '67', '68',
                '69', '7', '70',
                '71', '72', '73',
                '74', '75', '76',
                '77', '78', '79',
                '8', '80', '81',
                '82', '83', '84',
                '85', '86', '87',
                '88', '89', '9',
                '90', '91', '92',
                '93', '94', '95',
                '96', '97', '98',
                '99',
            ],
            'area': lon_lat,
        },
        '{}CAMS/{}/{}/{}/CAMS_{}.nc.zip'.format(path_save,datos,datetime_forecast.year,month,str(str_date).\
                                            replace(' ','_').replace(':','').replace('-','')))
    path_full = '{}CAMS/{}/{}/{}/CAMS_{}.nc.zip'.format(path_save,datos,datetime_forecast.year,month,str(str_date).\
                                            replace(' ','_').replace(':','').replace('-',''))
    print(path_full)
    print('/'.join(path_full.split('/')[:-1])+'/')
    os.system('unzip ' + path_full +' -d '+'/'.join(path_full.split('/')[:-1])+'/')
    os.system('mv '+'/'.join(path_full.split('/')[:-1])+'/data.nc '+path_full[:-4])
    os.system('rm ' + path_full +' -d '+'/'.join(path_full.split('/')[:-1])+'/')
    os.system(f'cp {path_full[:-4]} /var/data1/AQ_Forecast_DATA/operational/CAMS/')

    #Garantizamos que en la carpeta operacional hayan 15 archivos de dias
    lista = sorted(glob.glob('/var/data1/AQ_Forecast_DATA/operational/GFS/*CAMS*'))
    if len(lista)>15:
        os.remove(lista[0])

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
        date = pd.to_datetime(os.path.dirname(ruta_grib).split('/')[-1], format = '%Y%m%d') + \
            pd.Timedelta(os.path.basename(ruta_grib).split('t')[1][:2] + 'H')+\
            pd.Timedelta(os.path.basename(ruta_grib).split('f')[-1] + 'H')
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
        date = pd.to_datetime(os.path.dirname(ruta_grib).split('/')[-1], format = '%Y%m%d') + \
            pd.Timedelta(os.path.basename(ruta_grib).split('t')[1][:2] + 'H')+\
            pd.Timedelta(os.path.basename(ruta_grib).split('f')[-1] + 'H')
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


#Fechai = '2021-01-01'
#Fechaf ='2021-10-31'
#area = [12, -77, 2,-74]
#dataset = 'Reanalisis' #Pronostico o Reanalisis
#path_save = '/var/data1/AQ_Forecast_DATA/historic/'
#download_cams(Fechai,Fechaf, area, path_save, dataset)
