# -*- coding: utf-8 -*-

"""
Funciones de lectura de los archivos del ceilómetro, Radiómetro,Piranómetro
y Radar de Vientos
"""
import matplotlib
matplotlib.use("template")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
from matplotlib.colors import LogNorm
import os
import glob
import numpy as np
from datetime import datetime, timedelta
import psycopg2
import pandas as pd
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
#from windrose import WindroseAxes
import matplotlib.colors as colors
import MySQLdb

# ==========================================================================================
# Radiometro
# ==========================================================================================

def lee_data_ceil(ceilometro, Fecha_Inicio, Fecha_Fin):

   Fecha_1 = datetime.strftime(datetime.strptime(Fecha_Inicio,'%Y-%m-%d %H:%M:%S') + timedelta(hours=5),'%Y-%m-%d %H:%M:%S')
   Fecha_2 = datetime.strftime(datetime.strptime(Fecha_Fin,'%Y-%m-%d %H:%M:%S') + timedelta(hours=5) + timedelta(days=1),'%Y-%m-%d %H:%M:%S')

   File_List = pd.date_range(Fecha_1, Fecha_2, freq='1D')

   Backs  = []
   Fechas = []

   for idd, Fecha in enumerate(File_List):
       year  = datetime.strftime(Fecha, '%Y')
       month = datetime.strftime(Fecha, '%b')
       day   = datetime.strftime(Fecha, '%d')

       print (year+'-'+month+'-'+day + '   ' + ceilometro)

       fname  = '/mnt/ALMACENAMIENTO/ceilometro/datos/ceilometro'+str(ceilometro) \
                      +'/'+year+'/' + month+'/CEILOMETER_1_LEVEL_2_' +day+'.his'
       fname_ar  = '/mnt/ALMACENAMIENTO/ceilometro/datos/ceilometro'+str(ceilometro) \
                      +'/'+year+'/' + month

       archivo_ceilo= glob.glob(fname)
       if len(archivo_ceilo)==0:
        	os.system ('sshpass -p "mp960806" rsync -azt --bwlimit=500 -e ssh mariapvg@192.168.1.165:' + fname + " " + fname_ar)

       #print fname_miel
       #fname = fname_miel[-27:]
       print  (fname)

       try:
        BIN_fname  = np.genfromtxt(fname,delimiter=', ',dtype=object,usecols=(0,4),skip_header=2,\
                   converters = {0: lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S")})
        print ('11')

        DATA = np.array([decode_hex_string(BIN_fname[i,1]) for i in range(len(BIN_fname))]).T

        File_Dates = np.array(BIN_fname[:,0].tolist())
        print ('22')
       except:
          continue

       if idd == 0:
          Backs  = DATA
          Fechas = File_Dates
       else:
          Backs = np.concatenate([Backs,DATA],axis=1)
          Fechas = np.concatenate([Fechas, File_Dates])
   print (Backs)
   Backs = promedio(Backs, 3, 15)
   Backs = Backs.astype(np.float)
   Backs[Backs < 0] = np.NaN

   Backs = pd.DataFrame(Backs.T, np.array(Fechas) - timedelta(hours=5))
   Backs = Backs[Fecha_Inicio:Fecha_Fin]
   return Backs

def cloud_higth(Fecha_Inicio,Fecha_Fin, ceilometro):

	Fecha_Inicio = datetime.strftime(datetime.strptime(Fecha_Inicio, "%Y%m%d %H:%M") + timedelta(hours = 5),"%Y%m%d %H:%M")
	Fecha_Fin    = datetime.strftime(datetime.strptime(Fecha_Fin, "%Y%m%d %H:%M") + timedelta(hours = 5),"%Y%m%d %H:%M")

	File_List = pd.date_range(Fecha_Inicio[:8], Fecha_Fin[:8], freq='1D')
	Fechas   = []
	Cloud1   = []
	Cloud2   = []
	Cloud3   = []
	for idd, Fecha in enumerate(File_List):
		try:
			year  = datetime.strftime(Fecha, '%Y')
			month = datetime.strftime(Fecha, '%b')
			day   = datetime.strftime(Fecha, '%d')

			hisname  = '/mnt/ALMACENAMIENTO/ceilometro/datos/ceilometro'+str(ceilometro) \
			      +'/'+year+'/' + month+'/CEILOMETER_1_LEVEL_3_DEFAULT_' +day+'.his'
			print (hisname)
			Lista=np.genfromtxt(hisname,delimiter=', ',dtype=object,usecols=(0,-2,-4,-3),skip_header=1) 

			for j in range(1,len(Lista)):
				Fechas.append(Lista[j][0])
				Cloud1.append(np.float(Lista[j][1]))
				Cloud2.append(np.float(Lista[j][2]))
				Cloud3.append(np.float(Lista[j][3]))
		except:
			pass
		fechas=pd.to_datetime(Fechas)- timedelta(hours=5)
		Data = pd.DataFrame(index=fechas)
		Data["Cloud1"]=Cloud1
		Data["Cloud2"]=Cloud2
		Data["Cloud3"]=Cloud3
	Data[Data<0]=np.nan
#	Data1=Data.min(axis=1)
	#Data.index=pd.to_datetime(Data['0'])
	return Data


def Read_CSV(Dia):

    Dias = datetime.strptime(Dia,'%Y%m%d')

    # Fecha Inicio
    year  = datetime.strftime (Dias, "%Y")
    month = datetime.strftime (Dias, "%m")
    day   = datetime.strftime (Dias, "%d")

    # Lectura de Archivos según las fechas entregadas
#    PATH   = 'sshpass -p "mp960806" rsync -azt --bwlimit=500 -e ssh mariapvg@192.168.1.62:/mnt/ALMACENAMIENTO/radiometro/datos/'+year+'-'+month+'-'+day+'*_lv2.csv'
#    os.system('scp ' +PATH + ' ' + './')
    path_1   = '/mnt/ALMACENAMIENTO/radiometro/datos/'+year+'-'+month+'-'+day+'*_lv2.csv'
    filenames = sorted(glob.glob(path_1))


    lv2 =[0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90,
    1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25,
    3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 6.75, 7.00, 7.25,
    7.50, 7.75, 8.00, 8.25, 8.50, 8.75, 9.00, 9.25, 9.50, 9.75, 10.00]

    lv2 = np.array(lv2).astype(np.str).tolist()

    record = []
    date = []
    GPS = []
    ground = []
    cloud = []
    time = []

    Temperature_Zenit = []
    Temperature_15N = []
    Temperature_15S = []
    Temperature_15A = []

    VaporDensity_Zenit = []
    VaporDensity_15N = []
    VaporDensity_15S = []
    VaporDensity_15A = []

    Liquid_Zenit = []
    Liquid_15N = []
    Liquid_15S = []
    Liquid_15A = []

    RelativeHumidity_Zenit = []
    RelativeHumidity_15N = []
    RelativeHumidity_15S = []
    RelativeHumidity_15A = []

    for filename in filenames:

       print (filename)
       f = open(filename, "r" )

       for line in f:
          c = line.split('\r\n')
          a = c[0].split(',', 3)

          record.append(a[0])
          date.append(a[1])

          if a[2] == '31':
              GPS.append(a[3].split(','))

            #organizar la fecha de cada trama de datos
              t = a [1]
              time.append(datetime(2000+int(t[6:8]), int(t[0:2]), int(t[3:5]),
                                      int(t[9:11]), int(t[12:14]), int(t[15:17])))

          elif a[2] == '201':
            ground.append(a[3].split(','))

          elif a[2] == '301':
            cloud.append(a[3].split(',')[:-1])

          elif a[2] == '401':
            if a[3].startswith('ZenithKV'):
                b = a[3].split(',', 1)
                Temperature_Zenit.append(b[1].split(',')[:-1])

            if a[3].startswith('Angle15KV(N)'):
                b = a[3].split(',', 1)
                Temperature_15N.append(b[1].split(',')[:-1])

            if a[3].startswith('Angle15KV(S)'):
                b = a[3].split(',', 1)
                Temperature_15S.append(b[1].split(',')[:-1])

            if a[3].startswith('Angle15KV(A)'):
                b = a[3].split(',', 1)
                Temperature_15A.append(b[1].split(',')[:-1])

          elif a[2] == '402':
            if a[3].startswith('ZenithKV'):
                b = a[3].split(',', 1)
                VaporDensity_Zenit.append(b[1].split(',')[:-1])

            if a[3].startswith('Angle15KV(N)'):
                b = a[3].split(',', 1)
                VaporDensity_15N.append(b[1].split(',')[:-1])

            if a[3].startswith('Angle15KV(S)'):
                b = a[3].split(',', 1)
                VaporDensity_15S.append(b[1].split(',')[:-1])

            if a[3].startswith('Angle15KV(A)'):
                b = a[3].split(',', 1)
                VaporDensity_15A.append(b[1].split(',')[:-1])


          elif a[2] == '403':
            if a[3].startswith('ZenithKV'):
                b = a[3].split(',', 1)
                Liquid_Zenit.append(b[1].split(',')[:-1])

            if a[3].startswith('Angle15KV(N)'):
                b = a[3].split(',', 1)
                Liquid_15N.append(b[1].split(',')[:-1])

            if a[3].startswith('Angle15KV(S)'):
                b = a[3].split(',', 1)
                Liquid_15S.append(b[1].split(',')[:-1])

            if a[3].startswith('Angle15KV(A)'):
                b = a[3].split(',', 1)
                Liquid_15A.append(b[1].split(',')[:-1])


          elif a[2] == '404':
            if a[3].startswith('ZenithKV'):
                b = a[3].split(',', 1)
                RelativeHumidity_Zenit.append(b[1].split(',')[:-1])

            if a[3].startswith('Angle15KV(N)'):
                b = a[3].split(',', 1)
                RelativeHumidity_15N.append(b[1].split(',')[:-1])

            if a[3].startswith('Angle15KV(S)'):
                b = a[3].split(',', 1)
                RelativeHumidity_15S.append(b[1].split(',')[:-1])

            if a[3].startswith('Angle15KV(A)'):
                b = a[3].split(',', 1)
                RelativeHumidity_15A.append(b[1].split(',')[:-1])

    Temperature_Zenit = np.array(Temperature_Zenit).astype(np.float)
    Temperature_15N   = np.array(Temperature_15N).astype(np.float)
    Temperature_15S   = np.array(Temperature_15S).astype(np.float)
    Temperature_15A   = np.array(Temperature_15A).astype(np.float)

    RelativeHumidity_Zenit = np.array(RelativeHumidity_Zenit).astype(np.float)
    RelativeHumidity_15N   = np.array(RelativeHumidity_15N).astype(np.float)
    RelativeHumidity_15S   = np.array(RelativeHumidity_15S).astype(np.float)
    RelativeHumidity_15A   = np.array(RelativeHumidity_15A).astype(np.float)

    VaporDensity_Zenit = np.array(VaporDensity_Zenit).astype(np.float)
    VaporDensity_15N   = np.array(VaporDensity_15N).astype(np.float)
    VaporDensity_15S   = np.array(VaporDensity_15S).astype(np.float)
    VaporDensity_15A   = np.array(VaporDensity_15A).astype(np.float)

    Liquid_Zenit = np.array(Liquid_Zenit).astype(np.float)
    Liquid_15N   = np.array(Liquid_15N).astype(np.float)
    Liquid_15S   = np.array(Liquid_15S).astype(np.float)
    Liquid_15A   = np.array(Liquid_15A).astype(np.float)
 #   os.system('rm *_lv2.csv')
    # Data Frames
    # ==========================================================================
    a = 0

    while len(time) != len(Temperature_Zenit):
       if len(time)>len(Temperature_Zenit):
          del time[a]
       else:
          del Temperature_Zenit[a]

       if a == 0:
          a = -1
       else:
          a = 0
    if len(time) == len (Temperature_Zenit[:-1]):
       Temperature_Zenit      = pd.DataFrame(Temperature_Zenit[:-1],columns=np.array(lv2).astype(np.str).tolist(), index=time)
       RelativeHumidity_Zenit = pd.DataFrame(RelativeHumidity_Zenit[:-1],columns=np.array(lv2).astype(np.str).tolist(), index=time)
       VaporDensity_Zenit     = pd.DataFrame(VaporDensity_Zenit[:-1],columns=np.array(lv2).astype(np.str).tolist(), index=time)
       Liquid_Zenit           = pd.DataFrame(Liquid_Zenit[:-1],columns=np.array(lv2).astype(np.str).tolist(), index=time)
    else:
       Temperature_Zenit      = pd.DataFrame(Temperature_Zenit,columns=np.array(lv2).astype(np.str).tolist(), index=time)
       RelativeHumidity_Zenit = pd.DataFrame(RelativeHumidity_Zenit,columns=np.array(lv2).astype(np.str).tolist(), index=time)
       VaporDensity_Zenit     = pd.DataFrame(VaporDensity_Zenit,columns=np.array(lv2).astype(np.str).tolist(), index=time)
       Liquid_Zenit           = pd.DataFrame(Liquid_Zenit,columns=np.array(lv2).astype(np.str).tolist(), index=time)
    print ('comparo %d con %d'%(len(time),len (Temperature_15A[:-1])))
    if len(time) == len (Temperature_15A[:-1]):
       Temperature_15A       = pd.DataFrame(Temperature_15A[:-1],columns=np.array(lv2).astype(np.str).tolist(), index=time)
       RelativeHumidity_15A  = pd.DataFrame(RelativeHumidity_15A[:-1],columns=np.array(lv2).astype(np.str).tolist(), index=time)
       VaporDensity_15A      = pd.DataFrame(VaporDensity_15A[:-1],columns=np.array(lv2).astype(np.str).tolist(), index=time)
       Liquid_15A            = pd.DataFrame(Liquid_15A[:-1],columns=np.array(lv2).astype(np.str).tolist(), index=time)
    else:

       Temperature_15A       = pd.DataFrame(Temperature_15A,columns=np.array(lv2).astype(np.str).tolist(), index=time)
       RelativeHumidity_15A  = pd.DataFrame(RelativeHumidity_15A,columns=np.array(lv2).astype(np.str).tolist(), index=time)
       VaporDensity_15A      = pd.DataFrame(VaporDensity_15A,columns=np.array(lv2).astype(np.str).tolist(), index=time)
       Liquid_15A            = pd.DataFrame(Liquid_15A,columns=np.array(lv2).astype(np.str).tolist(), index=time)

    return  Temperature_Zenit, RelativeHumidity_Zenit, VaporDensity_Zenit, Liquid_Zenit, Temperature_15A, RelativeHumidity_15A, VaporDensity_15A, Liquid_15A

def lee_Radiometro(Fecha_Inicio, Fecha_Fin):

   print ('------------------------------------')
   print ('Lectura datos Radiometro')
   print ('------------------------------------')


   Fecha_Inicio = datetime.strftime(datetime.strptime(Fecha_Inicio, "%Y%m%d %H:%M") + timedelta(hours = 5),"%Y%m%d %H:%M")
   Fecha_Fin    = datetime.strftime(datetime.strptime(Fecha_Fin, "%Y%m%d %H:%M") + timedelta(hours = 5),"%Y%m%d %H:%M")

   Temperature_Zenit, RelativeHumidity_Zenit,VaporDensity_Zenit, Liquid_Zenit, Temperature_15A,\
     RelativeHumidity_15A, VaporDensity_15A, Liquid_15A  = Read_CSV(Fecha_Inicio[:8])

   Fechas = pd.date_range(Fecha_Inicio, Fecha_Fin, freq='1D')

   for i in Fechas[1:]:
       Dia = datetime.strftime(i, '%Y%m%d')
       print (Dia)
       try:
          Temp_Zenit, Hum_Zenit, Vap_Zenit, Liq_Zenit, Temp_15A,Hum_15A,Vap_15A,Liq_15A = Read_CSV(Dia)
          Temperature_Zenit      = Temperature_Zenit.append(Temp_Zenit)
          RelativeHumidity_Zenit = RelativeHumidity_Zenit.append(Hum_Zenit)
          Liquid_Zenit           = Liquid_Zenit.append(Liq_Zenit)
          VaporDensity_Zenit     = VaporDensity_Zenit.append(Vap_Zenit)

          Temperature_15A        = Temperature_15A.append(Temp_15A)
          RelativeHumidity_15A   = RelativeHumidity_15A.append(Hum_15A)
          Liquid_15A             = Liquid_15A.append(Liq_15A)
          VaporDensity_15A       = VaporDensity_15A.append(Vap_15A)

       except:
          pass

   lv2 =[0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90,
   1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25,
   3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 6.75, 7.00, 7.25,
   7.50, 7.75, 8.00, 8.25, 8.50, 8.75, 9.00, 9.25, 9.50, 9.75, 10.00]

   Liquid_Zenit = Liquid_Zenit[Fecha_Inicio:Fecha_Fin]
   Liquid_Zenit = pd.DataFrame(Liquid_Zenit.values, Liquid_Zenit.index-timedelta(hours=5))
   Liquid_Zenit.columns = lv2

   Temperature_Zenit = Temperature_Zenit[Fecha_Inicio:Fecha_Fin]
   Temperature_Zenit = pd.DataFrame(Temperature_Zenit.values, Temperature_Zenit.index-timedelta(hours=5))
   Temperature_Zenit.columns = lv2

   VaporDensity_Zenit = VaporDensity_Zenit[Fecha_Inicio:Fecha_Fin]
   VaporDensity_Zenit = pd.DataFrame(VaporDensity_Zenit.values, VaporDensity_Zenit.index-timedelta(hours=5))
   VaporDensity_Zenit.columns = lv2

   RelativeHumidity_Zenit = RelativeHumidity_Zenit[Fecha_Inicio:Fecha_Fin]
   RelativeHumidity_Zenit = pd.DataFrame(RelativeHumidity_Zenit.values, RelativeHumidity_Zenit.index-timedelta(hours=5))
   RelativeHumidity_Zenit.columns = lv2

   Liquid_15A = Liquid_15A[Fecha_Inicio:Fecha_Fin]
   Liquid_15A = pd.DataFrame(Liquid_15A.values, Liquid_15A.index-timedelta(hours=5))
   Liquid_15A.columns = lv2

   Temperature_15A = Temperature_15A[Fecha_Inicio:Fecha_Fin]
   Temperature_15A = pd.DataFrame(Temperature_15A.values, Temperature_15A.index-timedelta(hours=5))
   Temperature_15A.columns = lv2

   VaporDensity_15A = VaporDensity_15A[Fecha_Inicio:Fecha_Fin]
   VaporDensity_15A = pd.DataFrame(VaporDensity_15A.values, VaporDensity_15A.index-timedelta(hours=5))
   VaporDensity_15A.columns = lv2

   RelativeHumidity_15A = RelativeHumidity_15A[Fecha_Inicio:Fecha_Fin]
   RelativeHumidity_15A = pd.DataFrame(RelativeHumidity_15A.values, RelativeHumidity_15A.index-timedelta(hours=5))
   RelativeHumidity_15A.columns = lv2

   return Temperature_Zenit,RelativeHumidity_Zenit, Liquid_Zenit, VaporDensity_Zenit, Temperature_15A, RelativeHumidity_15A, VaporDensity_15A, Liquid_15A

# ==========================================================================================
# Piranometro
# ==========================================================================================


def lee_Piranometro(Fecha_Inicio, Fecha_Fin, Piranometro):

    print ('------------------------------------')
    print ('Lectura datos Piranometro')
    print ('------------------------------------')
    ### Piranometro= 'JoaquinVallejo' ,'BarbosaPdla' ,'AMVA' ,'itagui', 'siata'

    if Piranometro != 'siata':
       Fecha_Inicio = datetime.strftime(datetime.strptime(Fecha_Inicio, "%Y%m%d %H:%M") + timedelta(hours = 5),"%Y%m%d %H:%M")
       Fecha_Fin    = datetime.strftime(datetime.strptime(Fecha_Fin, "%Y%m%d %H:%M") + timedelta(hours = 5),"%Y%m%d %H:%M")


    File_List = pd.date_range(Fecha_Inicio[:8], Fecha_Fin[:8], freq='1D')

    Temp   = []
    Radiac = []
    Fechas = []

    for idd, Fecha in enumerate(File_List):
       Year  = datetime.strftime(Fecha, '%y')
       Month = datetime.strftime(Fecha, '%m')
       Day   = datetime.strftime(Fecha, '%d')
       #os.system('rm LOG*.csv')
       path  = "/mnt/ALMACENAMIENTO/piranometros/"+Piranometro+"/"
       name  = path + 'LOG'+ Year + Month + Day + '*.csv'
       #os.system ('sshpass -p "mp960806" rsync -azt --bwlimit=500 -e ssh mariapvg@192.168.1.165:' + name + " " + "./") #62

       files = glob.glob(name) #('LOG*.csv')

       Fecha=[]
       Rad=[]
       T=[]

       for file in files:
           print (file)
           Data  = np.genfromtxt(file, delimiter=';', dtype=np.str, skip_header=4, usecols= [1,2,5,6])
           for i in range(len(Data)):
                try:
                     Rad.append(np.float(Data[i][2]))
                     Fecha.append(datetime.strptime(Data[i][0]+' '+Data[i][1], '%Y-%m-%d %H:%M:%S')-timedelta(hours=5))
                     T.append(np.float(Data[i][3]) - 273.5)
                except: pass 
           # Concatena los arrays de los días seleccionados
           if idd == 0:
              Fechas = Fecha
              Radiac = Rad
              Temp   = T
           else:
              Fechas = np.concatenate([Fechas, Fecha], axis = 0)
              Radiac = np.concatenate([Radiac, Rad], axis = 0)
              Temp   = np.concatenate([Temp, T], axis = 0)

      # os.system('rm LOG*.csv')
      # for i in range(len(Data)):
      # 	try:
      # 		Rad   = np.array([np.float(Data[i][2])])
      # 		Fecha = np.array([datetime.strptime(Data[i][0]+' '+Data[i][1], '%Y-%m-%d %H:%M:%S')])-timedelta(hours=5)
#		T     = np.array([np.float(Data[i][3]) for i in range(len(Data))]) - 273.5
#       	except: pass



       # Concatena los arrays de los días seleccionados
       if idd == 0:
          Fechas = Fecha
          Radiac = Rad
          Temp   = T
       else:
          Fechas = np.concatenate([Fechas, Fecha], axis = 0)
          Radiac = np.concatenate([Radiac, Rad], axis = 0)
          Temp   = np.concatenate([Temp, T], axis = 0)

    Rad  = pd.DataFrame(Radiac, index=Fechas)
    #Temp = pd.DataFrame(Temp, Fechas)


    if Piranometro != 'siata':
       Fecha_Inicio = datetime.strftime(datetime.strptime(Fecha_Inicio, "%Y%m%d %H:%M") - timedelta(hours = 5),"%Y%m%d %H:%M")
       Fecha_Fin    = datetime.strftime(datetime.strptime(Fecha_Fin, "%Y%m%d %H:%M") - timedelta(hours = 5),"%Y%m%d %H:%M")
    try:
    	Rad =  Rad[(Rad.index>=Fecha_Inicio)&(Rad.index<=Fecha_Fin)]
    except: pass
    #Temp = Temp[Fecha_Inicio:Fecha_Fin]

    return Rad


# ==========================================================================================
# Ceilometro
# ==========================================================================================
def twos_comp(val, bits):
        if((val & (1 << (bits - 1))) != 0):
                 val = val - (1 << bits)
        return val

def decode_hex_string(string, fail_value=1, char_count=5, use_filter=False):
    data_len = len(string)
    data = np.zeros(data_len / char_count, dtype=int)
    key = 0
    for i in xrange(0, data_len, char_count):
        hex_string = string[i:i + char_count]
        data[key] = twos_comp(int(hex_string, 16), 20)
        key += 1
    if use_filter:
        data[data <= 0] = fail_value
        data = np.log10(data) - 9.
    return data

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    cdict = {'red': [],'green': [],'blue': [],'alpha': []}

    reg_index = np.linspace(start, stop, 257)
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

def promedio(DATA, minutos, metros):

   def mediamovil(interval, window_size):
      window = np.ones(int(window_size))/float(window_size)
      return np.convolve(interval, window, 'same')

   ventana = (minutos*60)/16
   metros = metros / 10.

   t_mean = []
   h_mean = []

   for i in range(DATA.shape[0]):
       t_mean.append(mediamovil(DATA[i],ventana))
   t_mean = np.array(t_mean)

   for i in range(DATA.shape[1]):
       h_mean.append(mediamovil(t_mean[:,i],metros))
   h_mean = np.array(h_mean).T

   return h_mean


def lee_Ceil(Fecha_Inicio,Fecha_Fin, ceilometro):

   print ('-------------------------------------------')
   print ('Lectura archivos Ceilometro')
   print ('-------------------------------------------')

   Fecha_Inicio = datetime.strftime(datetime.strptime(Fecha_Inicio, "%Y%m%d %H:%M") + timedelta(hours = 5),"%Y%m%d %H:%M")
   Fecha_Fin    = datetime.strftime(datetime.strptime(Fecha_Fin, "%Y%m%d %H:%M") + timedelta(hours = 5),"%Y%m%d %H:%M")

   File_List = pd.date_range(Fecha_Inicio[:8], Fecha_Fin[:8], freq='1D')

   Backs  = []
   Fechas = []

   for idd, Fecha in enumerate(File_List):
       year  = datetime.strftime(Fecha, '%Y')
       month = datetime.strftime(Fecha, '%b')
       day   = datetime.strftime(Fecha, '%d')

       fname_miel  = 'nroldan@192.168.1.62:/mnt/ALMACENAMIENTO/ceilometro/datos/ceilometro'+str(ceilometro) \
                      +'/'+year+'/' + month+'/CEILOMETER_1_LEVEL_2_' +day+'.his'


       os.system('scp ' + fname_miel + ' ' + './')
       fname = fname_miel[-27:]

       try:
          BIN_fname  = np.genfromtxt(fname,delimiter=', ',dtype=object,usecols=(0,4),skip_header=2,\
                   converters = {0: lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S")})

          DATA = np.array([decode_hex_string(BIN_fname[i,1]) for i in range(len(BIN_fname))]).T
          File_Dates = BIN_fname[:,0].tolist()

       except:
          DATA = np.ones([450,5397])   * np.NaN
          File_Dates = np.ones([5397]) * np.NaN


       # Concatena las matrices de los días seleccionados
       if idd == 0:
          Backs  = np.array(DATA)
          Fechas = np.array(File_Dates)
       else:
          Backs = np.concatenate([Backs,DATA],axis=1)
          Fechas = np.concatenate([Fechas, File_Dates])

   Backs = promedio(Backs, 3, 15) # Funcion que promedia en X minutos y Y metros
   Backs = Backs.astype(np.int)
   Backs[Backs < 100] = 100

   Backs = pd.DataFrame(Backs.T, Fechas - timedelta(hours=5))

   Fecha_Inicio = datetime.strftime(datetime.strptime(Fecha_Inicio, "%Y%m%d %H:%M") - timedelta(hours = 5),"%Y%m%d %H:%M")
   Fecha_Fin    = datetime.strftime(datetime.strptime(Fecha_Fin, "%Y%m%d %H:%M") - timedelta(hours = 5),"%Y%m%d %H:%M")

   Backs = Backs[Fecha_Inicio:Fecha_Fin]
   os.system('rm ' + fname_miel[-27:])
   return Backs

def gradiente(Bz):
    grad = np.array([(Bz[j+1]-Bz[j-1])/float(2.0) for j in range(len(Bz)-1)][1:])
    return grad

def cloudfilter_test(Bz, gradiente_Bz):
  if np.max(Bz) > 1200:
     min_grad = np.where(gradiente_Bz == np.min(gradiente_Bz))[0][0]
     max_grad = np.where(gradiente_Bz == np.max(gradiente_Bz))[0][0]
     try:
        base1 = 440

        base2 = np.where((gradiente_Bz[:max_grad][::-1] > -0.8) & (gradiente_Bz[:max_grad][::-1] < 0.8))[0][0]
        base2 = max_grad - base2

        if len(range(base2,base1,1)) > 10:
            print ('Nube')
            Bolean = True
        else:
            print ('No hay Nube')
            Bolean = False
        #m = (Bz[base2]-Bz[base1])/((base2)-(base1))

        #for i in range(base2,base1,1):
        #    Bz[i]= (m * (i- base2)) + Bz[base2]
     except:
        pass
     gradiente_Bz = gradiente(Bz)
  else:
     print ('No hay Nube')
     Bolean = False
  return Bolean

# ==========================================================================================
# Radar Perfilador de Vientos
# ==========================================================================================


def lee_RWP(Fecha_Inicio, Fecha_Fin,freq): # Freq = 5 ó 60

#    print ('-------------------------------------------')
#    print ('Lectura archivos Radar Perfilador de Viento')
#    print ('-------------------------------------------')


   Fecha_Inicio = datetime.strftime(datetime.strptime(Fecha_Inicio, "%Y%m%d %H:%M") + timedelta(hours = 5),"%Y%m%d %H:%M")
   Fecha_Fin    = datetime.strftime(datetime.strptime(Fecha_Fin, "%Y%m%d %H:%M") + timedelta(hours = 5),"%Y%m%d %H:%M")


   #Fecha Inicio
   year    = Fecha_Inicio[:4]
   month   = Fecha_Inicio[4:6]
   day     = Fecha_Inicio[6:8]
   hour    = Fecha_Inicio[9:11]
   minute  = Fecha_Inicio[12:14]

   # Ventana de Graficación
   year_2    = Fecha_Fin[:4]
   month_2   = Fecha_Fin[4:6]
   day_2     = Fecha_Fin[6:8]
   hour_2    = Fecha_Fin[9:11]
   minute_2  = Fecha_Fin[12:14]


   if freq == 60:

     Dia_1 = year + month + day + ' ' + hour + ':' + '00'
     Dia_2 = year_2 + month_2 +  day_2 + ' ' + hour_2 + ':' + '00'

     Fechas = pd.date_range(Dia_1, Dia_2,  freq = "1H")

   else:
     for i in range(5):
          minute = np.float(minute)
          if np.mod(minute , 5) == 0:
             minute = minute
             break
          else:
             minute = minute + 1

     Dia_1 = year + month + day + ' ' + hour + ':' + np.str(minute)[:-2]
     Dia_2 = year_2 + month_2 +  day_2 + ' ' + hour_2 + ':' + np.str(minute)[:-2]

     Fechas = pd.date_range(Dia_1, Dia_2,  freq = "5Min")

   files = []


   for i in Fechas:

       year    = datetime.strftime (i, "%Y")
       month   = datetime.strftime (i, "%m")
       day     = datetime.strftime (i, "%d")
       hour    = datetime.strftime (i, "%H")
       minute  = datetime.strftime (i, "%M")

       if freq == 60:
           path = '/mnt/ALMACENAMIENTO/RWP/'+year+'-'+month+'/w'+year+'-'+month+'-'+day+'-'+hour+'-00_60.asd'
       else:
          if int(month) < 5 and int(year) == 2015:
               path = '/mnt/ALMACENAMIENTO/RWP/'+year+'-'+month+'/w'+year+'-'+month+'-'+day+'-'+hour+'-'+minute+'_06.asd'
          else:
               path = '/mnt/ALMACENAMIENTO/RWP/'+year+'-'+month+'/w'+year+'-'+month+'-'+day+'-'+hour+'-'+minute+'_05.asd'

       files.append(path)


   Ventana_Horas  = int(((datetime.strptime(Dia_2,'%Y%m%d %H:%M') - datetime.strptime(Dia_1,'%Y%m%d %H:%M')).total_seconds())/(60.* 60.))
   Ventana_Minuto = int(((datetime.strptime(Dia_2,'%Y%m%d %H:%M') - datetime.strptime(Dia_1,'%Y%m%d %H:%M')).total_seconds())/(60.))


   if freq == 60:
   # Matrices para analizar datos cada Hora. Ventana está en días
      Altura     = np.ones([134,(Ventana_Horas) + 1]) * np.NaN
      Velocidad  = np.ones([134,(Ventana_Horas) + 1]) * np.NaN
      Direccion  = np.ones([134,(Ventana_Horas) + 1]) * np.NaN
      Calidad    = np.ones([134,(Ventana_Horas) + 1]) * np.NaN
      Zonal      = np.ones([134,(Ventana_Horas) + 1]) * np.NaN
      Meridional = np.ones([134,(Ventana_Horas) + 1]) * np.NaN
      Omega      = np.ones([134,(Ventana_Horas) + 1]) * np.NaN

   else:
   # Matrices para analizar datos cada 5 minutos
      Altura     = np.ones([134,(Ventana_Minuto)/5 + 1]) * np.NaN
      Velocidad  = np.ones([134,(Ventana_Minuto)/5 + 1]) * np.NaN
      Direccion  = np.ones([134,(Ventana_Minuto)/5 + 1]) * np.NaN
      Calidad    = np.ones([134,(Ventana_Minuto)/5 + 1]) * np.NaN
      Zonal      = np.ones([134,(Ventana_Minuto)/5 + 1]) * np.NaN
      Meridional = np.ones([134,(Ventana_Minuto)/5 + 1]) * np.NaN
      Omega      = np.ones([134,(Ventana_Minuto)/5 + 1]) * np.NaN

   for i in range(len(Fechas)):
    if i%200 == 0:
        print(Fechas[i])
    for j in range(len(files)):
      if datetime.strftime(Fechas[i], '%Y-%m-%d-%H-%M') == files[j][33:49]:
#          print (Fechas[i])
#          print (files[j])
         try:

            #os.system('sshpass -p "mp960806" rsync -azt --bwlimit=500 -e ssh mariapvg@192.168.1.62:'+ files[j] + ' ./')
            Data_mode1 = np.genfromtxt(files[j],dtype=np.float,skip_header=9,skip_footer=88, usecols = (0,1,2,3,4,5,6,7))
            Data_mode2 = np.genfromtxt(files[j],dtype=np.float,skip_header=76,skip_footer=1, usecols = (0,1,2,3,4,5,6,7))
            #os.system('rm *.asd')

            #Data_mode1 = np.genfromtxt(files[j],dtype=np.float,skip_header=9,skip_footer=88, usecols = (0,1,2,3,4,5,6,7))
            #Data_mode2 = np.genfromtxt(files[j],dtype=np.float,skip_header=76,skip_footer=1, usecols = (0,1,2,3,4,5,6,7))

            Altura[:57,i]     = Data_mode1[:,0]
            #Velocidad[:57,i]  = Data_mode1[:,1]
            #Direccion[:57,i]  = Data_mode1[:,2]
            #Calidad[:57,i]    = Data_mode1[:,3]
            Zonal[:57,i]      = Data_mode1[:,4]
            Meridional[:57,i] = Data_mode1[:,5]
            Omega[:57,i]      = Data_mode1[:,6]


            Altura[57:,i]     = Data_mode2[:,0]
            #Velocidad[57:,i]  = Data_mode2[:,1]
            #Direccion[57:,i]  = Data_mode2[:,2]
            #Calidad[57:,i]    = Data_mode2[:,3]
            Zonal[57:,i]      = Data_mode2[:,4]
            Meridional[57:,i] = Data_mode2[:,5]
            Omega[57:,i]      = Data_mode2[:,6]
         except:
           pass

      else:
         pass


   Fechas = Fechas - timedelta (hours = 5)

   idx       = np.argsort(Altura[:,0])
   Altura    = Altura[idx]
   #Velocidad = Velocidad[idx]
   #Direccion = Direccion[idx]
   Zonal     = Zonal[idx]
   Meridional= Meridional[idx]
   Omega     = Omega[idx]

   Zonal[Zonal > 999]           = np.NaN
   Meridional[Meridional > 999] = np.NaN
   Velocidad[Velocidad > 999]   = np.NaN
   Omega[Omega > 999]           = np.NaN
   Direccion[Direccion > 999]   = np.NaN

   Zonal      = pd.DataFrame(Zonal.T, index = Fechas)
   Meridional = pd.DataFrame(Meridional.T,  index = Fechas)
   Altura  = pd.DataFrame(Altura.T,  index = Fechas)
   #Velocidad  = pd.DataFrame(Velocidad.T,  index = Fechas)
   Omega      = pd.DataFrame(Omega.T, index = Fechas)
   #Direccion  = pd.DataFrame(Direccion.T, index = Fechas)
   return  Omega ,Zonal, Meridional,Altura
   #return Velocidad , Direccion, Omega ,Altura, Fechas, Velocidad, Direccion, Zonal, Meridional, Omega


# ==========================================================================================
# Estaciones Calidad del aire
# ==========================================================================================

def Aire_db(Variable, Estaciones, Fecha_Inicio, Fecha_Fin):

  print ('-------------------------------------------')
  print ('Lectura datos Estaciones Redaire')
  print ('-------------------------------------------')

  print (Variable)
  print ('Estacion = '+ np.str(Estaciones[0]))
  print ('\n')

  host   = "192.168.1.74"
#  user   = "usrCalidad"
  user   = "siataRedAire_C"
#  passwd = "aF05wnXC;"
  passwd = "CalidadA1r3_C0nsult4"
  dbname = "siataRedAire"

  Data_dict = {}

  for Estacion in Estaciones:

    if (Variable == 'PM2.5' and Estacion == 48):

         query_1 = "SELECT fecha_hora,valor FROM redAireData WHERE CodigoSerial="+str(Estacion)+" AND fecha_hora BETWEEN \
           '"+np.str(Fecha_Inicio)+"' AND '"+ np.str(Fecha_Fin)+"' AND variable_name = '" +str(Variable)+ "' order by fecha_hora"

         query_2 = "SELECT fecha_hora,valor FROM redAireData WHERE CodigoSerial="+str(Estacion)+" AND fecha_hora BETWEEN \
           '"+np.str(Fecha_Inicio)+"' AND '"+ np.str(Fecha_Fin)+"' AND variable_name = 'TAire10_SSR' order by fecha_hora"

    elif (Variable == 'PM2.5' and Estacion == 12):

         query_1 = "SELECT fecha_hora,valor FROM redAireData WHERE CodigoSerial="+str(Estacion)+" AND fecha_hora BETWEEN \
            '"+np.str(Fecha_Inicio)+"' AND '"+ np.str(Fecha_Fin)+"' AND variable_name= '" +str(Variable)+ "_' order by fecha_hora"

    elif (Variable == 'PM10' and Estacion == 12):

         query_1 = "SELECT fecha_hora,"+Variable+" FROM redAireData WHERE CodigoSerial="+str(Estacion)+" AND fecha_hora BETWEEN \
            '"+np.str(Fecha_Inicio)+"' AND '"+ np.str(Fecha_Fin)+"' AND variable_name= '" +str(Variable)+ "_' order by fecha_hora"

    else :
         var='pm25' if Variable == 'PM2.5' else Variable
         query_1 = "SELECT fecha_hora,"+var+" FROM redAireDataValidadoSIATA WHERE CodigoSerial="+str(Estacion)+" AND fecha_hora BETWEEN \
            '"+np.str(Fecha_Inicio)+"' AND '"+ np.str(Fecha_Fin)+"' AND calidad_"+var+"<2.6 order by fecha_hora"

    print (query_1)
    # Conexion
    conn_db = MySQLdb.connect (host, user, passwd, dbname)

    # Consulta
    db_cursor = conn_db.cursor ()
    db_cursor.execute (query_1)
    Aire_Data = db_cursor.fetchall ()

    Data   = []
    Fechas = []

    for j in range(len(Aire_Data)):
      Data.append(np.float(Aire_Data[j][1]))
      Fechas.append(Aire_Data[j][0])

    DataFrame = pd.DataFrame(Data,Fechas)

    # Filtra Negativos y Faltantes 985
    DataFrame.values[DataFrame.values >800] = np.NaN
    DataFrame.values[DataFrame.values <  0] = np.NaN


    if (Variable == 'PM2.5' and Estacion == 48):
      db_cursor.execute (query_2)
      Temp_Data = db_cursor.fetchall ()

      Fechas_Temp = []
      Temp   = []
      for j in range(len(Aire_Data)):
         Fechas_Temp.append(Temp_Data[j][0])
         Temp.append(np.float(Temp_Data[j][1]))

      DataFrame_Temp = pd.DataFrame(Temp,Fechas_Temp)
      DataFrame_Temp = DataFrame_Temp.reindex(Fechas)
      DataFrame_Temp.values[DataFrame_Temp.values <  0] = np.NaN

      DataFrame = (760.0/(628.4*298))*(DataFrame_Temp+273.15)* DataFrame

    Data_dict[Estacion] = DataFrame

    conn_db.close ()

  return Data_dict



def Aire_Validados(Variable, Estaciones, Fecha_Inicio, Fecha_Fin):

  print ('-------------------------------------------')
  print ('Lectura datos Aire Validados')
  print ('-------------------------------------------')

  print (Variable)
  print ('Estacion = '+ np.str(Estaciones[0]))
  print ('\n')


  host   = "192.168.1.74"
  user   = "siataRedAire_C"
  passwd = "CalidadA1r3_C0nsult4"
  dbname = "siataRedAire"

  Data_dict = {}

  for Estacion in Estaciones:


    query_1 = "SELECT fecha_hora, "+str(Variable)+", CAST(calidad_" + str(Variable) + " AS DECIMAL (25,15)) FROM redAireDataValidadoSIATA WHERE CodigoSerial="+str(Estacion)+" AND fecha_hora BETWEEN \
           '"+np.str(Fecha_Inicio)+"' AND '"+ np.str(Fecha_Fin)+"' AND CAST(calidad_" + str(Variable) + " AS DECIMAL (25,15)) < 2.6 order by fecha_hora"


    # Conexion
    conn_db = MySQLdb.connect (host, user, passwd, dbname)

    # Consulta
    db_cursor = conn_db.cursor ()
    db_cursor.execute (query_1)
    Aire_Data = db_cursor.fetchall ()

    Data   = []
    Fechas = []

    for j in range(len(Aire_Data)):
      Data.append(np.float(Aire_Data[j][1]))
      Fechas.append(Aire_Data[j][0])

    DataFrame = pd.DataFrame(Data,Fechas)

    # Filtra Negativos y Faltantes 985
    DataFrame.values[DataFrame.values >800] = np.NaN
    DataFrame.values[DataFrame.values <  0] = np.NaN
#    DataFrame = DataFrame.reindex(pd.date_range(Fecha_Inicio, Fecha_Fin, freq='1H'))

    Data_dict[Estacion] = DataFrame

    conn_db.close ()

  return Data_dict

def Aire_Manual(Variable, Estaciones, Fecha_Inicio, Fecha_Fin):

  print ('-------------------------------------------')
  print ('Lectura datos Aire Validados')
  print ('-------------------------------------------')

  print (Variable)
  print ('Estacion = '+ np.str(Estaciones[0]))
  print ('\n')

  host   = "192.168.1.74"
  user   = "siataRedAire_C"
  passwd = "CalidadA1r3_C0nsult4"
  dbname = "siataRedAire"

  Data_dict = {}

  for Estacion in Estaciones:

    query_1 = "SELECT fecha_hora, " + str(Variable) +" FROM redAireDataManual WHERE nombreCorto ='"+ str(Estacion)+"' AND fecha_hora BETWEEN '"+np.str(Fecha_Inicio)+"' AND '"+ np.str(Fecha_Fin)+"'  order by fecha_hora"

    # Conexion
    conn_db = MySQLdb.connect (host, user, passwd, dbname)

    # Consulta
    db_cursor = conn_db.cursor ()
    db_cursor.execute (query_1)
    Aire_Data = db_cursor.fetchall ()

    Data   = []
    Fechas = []

    for j in range(len(Aire_Data)):
      Data.append(np.float(Aire_Data[j][1]))
      Fechas.append(Aire_Data[j][0])

    DataFrame = pd.DataFrame(Data,Fechas)

    # Filtra Negativos y Faltantes 985
    DataFrame.values[DataFrame.values >800] = np.NaN
    DataFrame.values[DataFrame.values <  0] = np.NaN

    Data_dict[Estacion] = DataFrame

    conn_db.close ()

  return Data_dict

def Aire_ValidadosSIATA(Variable, Estaciones, Fecha_Inicio, Fecha_Fin):

  print ('-------------------------------------------')
  print ('Lectura datos Aire Validados')
  print ('-------------------------------------------')

  Variable = Variable.lower ()
  Variable = Variable.replace (".", "")

  print (Variable)
  print ('Estacion = '+ np.str(Estaciones[0]))
  print ('\n')

  host   = "192.168.1.74"
  user   = "siataRedAire_C"
  passwd = "CalidadA1r3_C0nsult4"
  dbname = "siataRedAire"

  Data_dict = {}

  for Estacion in Estaciones:


    query_1 = "SELECT fecha_hora, "+str(Variable)+", CAST(calidad_" + str(Variable) + " AS DECIMAL (25,15)) FROM redAireDataValidadoSIATA WHERE CodigoSerial="+str(Estacion)+" AND fecha_hora BETWEEN \
           '"+np.str(Fecha_Inicio)+"' AND '"+ np.str(Fecha_Fin)+"' AND CAST(calidad_" + str(Variable) + " AS DECIMAL (25,15)) < 2.6 order by fecha_hora"

    # Conexion
    conn_db = MySQLdb.connect (host, user, passwd, dbname)

    # Consulta
    db_cursor = conn_db.cursor ()
    db_cursor.execute (query_1)
    Aire_Data = db_cursor.fetchall ()

    Data   = []
    Fechas = []

    for j in range(len(Aire_Data)):
      Data.append(np.float(Aire_Data[j][1]))
      Fechas.append(Aire_Data[j][0])

    DataFrame = pd.DataFrame(Data,Fechas)

    # Filtra Negativos y Faltantes 985
    DataFrame.values[DataFrame.values >800] = np.NaN
    DataFrame.values[DataFrame.values <  0] = np.NaN

    Data_dict[Estacion] = DataFrame

    conn_db.close ()

  return Data_dict



def Meteo(Estacion, var, Fecha_1, Fecha_2,Equipo='Thies'):

  host   = "192.168.1.100"
#  user   = "usrCalidad"
  user   = "siata_Consulta"
#  passwd = "aF05wnXC;"
  passwd = "si@t@64512_C0nsult4"
  dbname = "siata"
  Analoga={'P_SSR':['pr','pa'],'PLiquida_SSR':['p','rc'],'TAire10_SSR':['t','ta'],'HAire10_SSR':['h','ua'],'VViento_SSR':['vv','sm'],'DViento_SSR':['dv','dm']}
  calidad_siata={'TAire10_SSR':[153,1513,1534,1535,1536,15134,15135,15136,15345,15346,15356,151,253,2513,2534,2535,2536,25134,25135,25136,25345,25346,25356,251],
		'PLiquida_SSR':[1511,1513,1514,1515,1516,15134,15135,15136,15145,15146,15156,151,2511,2513,2514,2515,2516,25134,25135,25136,25145,25146,25156,251],
		'P_SSR':[155,1515,1535,1545,1556,15135,15145,15156,15345,15356,15456,151,255,2515,2535,2545,2556,25135,25145,25156,25345,25356,25456,251],
		'HAire10_SSR':[154,1514,1534,1545,1546,15134,15145,15146,15345,15346,15356,151,254,2514,2534,2545,2546,25134,25145,25146,25345,25346,25356,251],
		'VViento_SSR':[156,1516,1536,1546,1556,15136,15146,15156,15346,15456,15356,151,256,2516,2536,2546,2556,25136,25146,25156,25346,25456,25356,251],
		'DViento_SSR':[156,1516,1536,1546,1556,15136,15146,15156,15346,15456,15356,151,256,2516,2536,2546,2556,25136,25146,25156,25346,25456,25356,251]}
  Calculovaisala={'P_SSR':10.,'PLiquida_SSR':1000.,'TAire10_SSR':10.,'HAire10_SSR':10.,'VViento_SSR':10.,'DViento_SSR':1.}
  if Equipo=='Thies':
  	query = "SELECT fecha_hora,"+Analoga[var][0]+" FROM meteo_thiess WHERE cliente="+str(Estacion)+" AND  "+Analoga[var][0]+"<> -999. AND calidad not in ("+str(calidad_siata[var]).strip("[]")+") AND fecha_hora BETWEEN '"+np.str(Fecha_1)+"' AND '"+np.str(Fecha_2)+"' order by fecha_hora"
  if Equipo=='Vaisala':
  	query = "SELECT TIMESTAMP(fecha,hora),"+Analoga[var][1]+" FROM vaisala WHERE cliente="+str(Estacion)+" AND calidad not in ("+str(calidad_siata[var]).strip("[]")+") AND fecha_hora BETWEEN '"+np.str(Fecha_1)+"' AND '"+np.str(Fecha_2)+"' order by fecha_hora"
  print (query)
  conn_db = MySQLdb.connect (host, user, passwd, dbname)

  db_cursor = conn_db.cursor ()
  db_cursor.execute (query)

  Data = db_cursor.fetchall ()

  fecha=[]
  datos=[]

  for j in range(len(Data)):
       fecha.append(Data[j][0])
       if Equipo=='Vaisala':
            datos.append(Data[j][1]/Calculovaisala[var])
       elif Equipo=='Thies':
            datos.append(Data[j][1])
  Solicitud = pd.DataFrame({var:datos}, index=fecha)

  return Solicitud

def Vaisala(Estacion, Fecha_1, Fecha_2):

  host   = "192.168.1.74"
#  user   = "usrCalidad"
  user   = "siata_Consulta"
#  passwd = "aF05wnXC;"
  passwd = "si@t@64512_C0nsult4"
  dbname = "siata"

  query = "SELECT TIMESTAMP(fecha,hora),ta,Ua,rc,Sm,Dm,Sx,Dx,Pa FROM vaisala WHERE cliente="+str(Estacion)+" AND TIMESTAMP(fecha,hora) BETWEEN '"+np.str(Fecha_1)+"' AND '"+np.str(Fecha_2)+"' order by TIMESTAMP(fecha,hora)"
  #query = "SELECT TIMESTAMP(fecha,hora),ta-avg(ta),Ua-avg(Ua),Sm-avg(Sm),Dm-avg(Dm),Pa-avg(Pa) FROM vaisala WHERE cliente="+str(Estacion)+"AND rc=0.0 AND TIMESTAMP(fecha,hora) BETWEEN '"+np.str(Fecha_1)+"' AND '"+np.str(Fecha_2)+"' order by TIMESTAMP(fecha,hora)"
  conn_db = MySQLdb.connect (host, user, passwd, dbname)
  print (query)

  db_cursor = conn_db.cursor ()
  db_cursor.execute (query)

  Data = db_cursor.fetchall ()

  Fechas = []
  Temp   = []
  Hum    = []
  #Precip = []
  Speed  = []
  Direc  = []
  #none   = []
  #Speed_max = []
  #Direc_max = []
  Humedad   = []
  Presion   = []

  for j in range(len(Data)):
      if (Data[j][0] != None) and (Data[j][1] != None) and (Data[j][2] != None) and (Data[j][3] != None) and (Data[j][4] != None) and (Data[j][5] != None): #and (Data[j][6] != None) and (Data[j][7] != None):
           Fechas.append(Data[j][0].replace(second = 0))
           Temp.append(np.float(Data[j][1])/10.)
           Hum.append(np.float(Data[j][2])/10.)
           #Precip.append(np.float(Data[j][3])/100.)
           Speed.append(np.float(Data[j][3])/10.)
           Direc.append(np.float(Data[j][4]))
           #Speed_max.append(np.float(Data[j][5])/10.)
           #Direc_max.append(np.float(Data[j][6]))
           Presion.append(np.float(Data[j][5])/10.)

  conn_db.close ()

  Humedad.append(Hum)
  Speed = np.array(Speed)
  Direc = np.array(Direc)

  Speed[Speed < 0] = np.NaN
  Direc[Direc < 0] = np.NaN
  Humedad[Humedad < 0 ] = np.NaN

  #Vaisala = pd.DataFrame({'Speed_Mean':Speed, 'Dir_Mean':Direc ,'Temp':Temp, 'Hum':Hum}, index=Fechas)
  Vaisala = pd.DataFrame({'Speed_Mean':Speed, 'Dir_Mean':Direc, 'Temp':Temp, 'Hum':Hum, 'presion':Presion}, index=Fechas)
  #Vaisala = pd.DataFrame({'Speed_Mean':Speed, 'Dir_Mean':Direc, 'Speed_Max':Speed_max, 'Direc_max':Direc_max}, index=Fechas)

  #Temp      = pd.DataFrame(Temp,Fechas)
  #Hum       = pd.DataFrame(Hum,Fechas)
  #Precip    = pd.DataFrame(Precip,Fechas)
  #Speed     = pd.DataFrame(Speed, Fechas)
  #Direc     = pd.DataFrame(Direc, Fechas)
  #Speed_max = pd.DataFrame(Speed_max, Fechas)
  #Direc_max = pd.DataFrame(Direc_max, Fechas)
  #Presion   = pd.DataFrame(Presion, Fechas)

  return Vaisala

def Meteo_Thiess(Variable, Estaciones, Fecha_Inicio, Fecha_Fin):

  print ('-------------------------------------------')
  print ('Lectura Datos Meteo Thiess')
  print ('-------------------------------------------')

  host   = "192.168.1.74"
  user   = "siata_Consulta"
  passwd = "si@t@64512_C0nsult4"
  dbname = "siata"

  Data_dict = {}

  for Estacion in Estaciones:

    print (Variable)
    print ('Estacion = '+ np.str(Estacion))
    print ('\n')

    query_1 = "SELECT fecha_hora, "+str(Variable)+", calidad FROM meteo_thiess WHERE cliente="+str(Estacion)+" AND fecha_hora BETWEEN " +\
           "'"+np.str(Fecha_Inicio)+"' AND '"+ np.str(Fecha_Fin)+"' AND calidad = 1 order by fecha_hora"
    print (query_1)
    # Conexion
    conn_db = MySQLdb.connect (host, user, passwd, dbname)

    # Consulta
    db_cursor = conn_db.cursor ()
    db_cursor.execute (query_1)
    Aire_Data = db_cursor.fetchall ()

    Data   = []
    Fechas = []

    for j in range(len(Aire_Data)):
      Data.append(np.float(Aire_Data[j][1]))
      Fechas.append(Aire_Data[j][0])

    DataFrame = pd.DataFrame(Data,Fechas)

    # Filtra Negativos y Faltantes 985
#    DataFrame.values[DataFrame.values >800] = np.NaN
#    DataFrame.values[DataFrame.values <  0] = np.NaN
#    DataFrame = DataFrame.reindex(pd.date_range(Fecha_Inicio, Fecha_Fin, freq='1H'))

    Data_dict[Estacion] = DataFrame

    conn_db.close ()

  return Data_dict

def Turbulencia(Fecha_Inicio, Fecha_Fin,Crudos=True):

     File_List = pd.date_range(Fecha_Inicio[:10], Fecha_Fin[:10], freq='1D')

     for idd, Fecha in enumerate(File_List):
         print (Fecha)
         Year  = datetime.strftime(Fecha, '%Y')
         Month = datetime.strftime(Fecha, '%m')
         Day   = datetime.strftime(Fecha, '%d')

         path = '/var/data1/DatosTurbulencia/IRGASON_Federico_Carrasquilla/Datos_Flux/'
         if Crudos:
            file = 'Flux_Notes_' + Year +'-'+ Month +'-'+Day + '.csv'
         else:
            file = 'Flux_CSIFormat_' + Year +'-'+ Month +'-'+Day + '.csv'

         Data = np.genfromtxt(path + file, delimiter=',', dtype='str', skip_header = 1 )
         Data[Data == '"NAN"'] = np.NaN
         Data[Data == '"Grass"'] = np.NaN
         Data[Data == '"Kljun et al"'] = np.NaN
         Data[Data == '"KormannMeixner"'] = np.NaN

         Fechas = [datetime.strptime(Data[i,0],'"%Y-%m-%d %H:%M:%S"') for i in range(3, len(Data))]
         Flujos_Turb = {}

         for Variable in range(2, Data.shape[1]):
             Flujos_Turb[str(Data[0,Variable][1:-1])] = Data[3:,Variable].astype(np.float)


         Flujos_Turb = pd.DataFrame(Flujos_Turb, index = Fechas)


         # Concatena los arrays de los días seleccionados
         if idd == 0:
             Flujos = Flujos_Turb
         else:
             Flujos= Flujos.append(Flujos_Turb)


         Flujos = Flujos[Fecha_Inicio:Fecha_Fin]

     return Flujos
