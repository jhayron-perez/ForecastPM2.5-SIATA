#!/usr/bin/env python
# coding: utf-8
import datetime
from datetime import datetime
import pandas as pd
import sys
sys.path.insert(1,'/home/jsperezc/jupyter/AQ_Forecast/functions/')
from data_downloading import download_cams_forecast

current_day_time = datetime.now() 
current_time = current_day_time.time() #Extraemos solo la hora 
current_day = current_day_time.date() #Extraemos solo la fecha 
if (datetime.strptime('05:00:00','%H:%M:%S').time() < current_time <
     datetime.strptime('17:00:00','%H:%M:%S').time()):
    hour = 0
elif (datetime.strptime('17:00:00','%H:%M:%S').time() < current_time <
      datetime.strptime('23:59:59','%H:%M:%S').time()
      or datetime.strptime('00:00:00','%H:%M:%S').time() < current_time <
      datetime.strptime('05:00:00','%H:%M:%S').time()):
    hour = 12

Fechai = datetime(current_day.year,current_day.month,current_day.day,hour) 
area = [12, -77, 2,-74]
path_save = '/var/data1/AQ_Forecast_DATA/historic/'

#Fec_Ini ='2021-12-29'
#Fec_Fin = '2021-12-29'
#hour = [0,12]

#==================================================================================
#Descarga dia especifico
#Fechai = datetime.datetime(2021,12,7,0)
#area = [12, -77, 2,-74]
#path_save = '/var/data1/AQ_Forecast_DATA/historic/'
#download_cams_forecast(Fechai, area, path_save)
#==================================================================================

#==================================================================================
#Descargar Varios dias o meses -> OJO con variables iniciales de arriba del codigo
#fecs = pd.date_range(Fec_Ini, Fec_Fin, freq = 'D')

#for i in fecs:
#    for j in hour:
#        Fechai = datetime(i.year,i.month,i.day,j)
#        area = [12, -77, 2,-74]
#        path_save = '/var/data1/AQ_Forecast_DATA/historic/'
#        download_cams_forecast(Fechai, area, path_save)

download_cams_forecast(Fechai, area, path_save)
