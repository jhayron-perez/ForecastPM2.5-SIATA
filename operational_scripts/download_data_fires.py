# This Python file uses the following encoding: utf-8

import os

os.chdir('/var/data1/AQ_Forecast_DATA/operational/Fires/')
os.system('curl -O -k https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_South_America_7d.csv')
