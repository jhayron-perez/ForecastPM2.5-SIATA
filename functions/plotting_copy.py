def plot_forecasts_24h_individual_operational(name_station,forecasts,pm2p5,path_output,
    probabilities=False,df_probs=None):
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    import matplotlib
    import datetime as dt
    
    ### FIGURA ICA:
    pm2p5_24h_mean = pm2p5.rolling(24).mean()

    limites_ica = {'C.A. Buena':(0,12.5),
                  'C.A. Moderada':(12.5,37.5),
                  'C.A. Dañina a grupos sensibles':(37.5,55.5),
                  'C.A. Dañina a la salud':(55.5,150.5),
                  'C.A. Muy dañina a la salud':(150.5,250.5),
                  'C.A. Peligrosa':(250.5,500)}

    colores_ica = ['green','yellow','orange','red','purple','brown']

    fig = plt.figure(figsize=(11,5))
    # fig.subplots_adjust(left=0.1, wspace=0.1)
    # plt.subplot2grid((1, 16), (0, 0), colspan=12)

    ## GRAPH
    dates_forecast = forecasts.index
    for model_name in forecasts.keys():
        if model_name == 'LR_RC':
            plt.plot(dates_forecast,forecasts[model_name].values,label = 'LR_CH'.replace('_','-')\
                     ,ls='--',alpha=1)
        elif model_name == 'GB_MO':
            plt.plot(dates_forecast,forecasts[model_name].values,label = model_name.replace('_','-')\
                     +' (mejor modelo)',alpha=1,lw=2)
        else:
            plt.plot(dates_forecast,forecasts[model_name].values,label = model_name.replace('_','-')\
                     ,alpha=1)
    
    forecasts['mean'] = np.mean(forecasts,axis = 1)
    plt.plot(forecasts['mean'],label = 'Media del ensamble',
             color='teal',lw=2)
    
    if probabilities==False:
        plt.fill_between(dates_forecast,np.min(forecasts,axis = 1),np.max(forecasts,axis = 1),
                        alpha=0.35,color='skyblue',label='Rango del ensamble')
    else:
        df_probs = df_probs.rolling(3,min_periods=1,center=True).mean()
        plt.plot(df_probs['p10'],color='k',alpha=0.1,ls='--')
        plt.plot(df_probs['p90'],color='k',alpha=0.1,ls='--')
        plt.plot(df_probs['p25'],color='k',alpha=0.1,ls='--')
        plt.plot(df_probs['p75'],color='k',alpha=0.1,ls='--')
        plt.fill_between(dates_forecast,df_probs['p10'],df_probs['p90'],
                        alpha=0.35,color='skyblue',label='Percentil 10-90',ls='--')
        plt.fill_between(dates_forecast,df_probs['p25'],df_probs['p75'],
                        alpha=0.6,color='skyblue',label='Rango intercuartil',ls='--')
    forecast_initial_date = dates_forecast[0]-dt.timedelta(hours=1)
    plt.axvline(forecast_initial_date,color='k',ls = '--',label='Fecha inicial de pronóstico')
    plt.plot(pm2p5_24h_mean,lw=1.4,color='k',label='Observaciones')

    plt.xticks(fontsize=(12))
    plt.yticks(fontsize=(12))
    plt.ylabel('Concentración de PM2.5\n(promedios de 24 horas)\n[$\mu g/m^3$]',fontsize=14)
    plt.xlim(dates_forecast[0]-dt.timedelta(hours=48),dates_forecast[-1])
    plt.grid(ls='--',alpha=0.3)
    plt.title(name_station+' - Fecha inicial de pronóstico '+str(forecast_initial_date)+ ' HL',\
              fontsize=15,loc='left',y = 1.02)

    #### Background ICA

    for i in range(len(colores_ica)):
        plt.fill_between([dates_forecast[0]-dt.timedelta(hours=48),dates_forecast[-1]],
                    limites_ica[list(limites_ica.keys())[i]][0],
                    limites_ica[list(limites_ica.keys())[i]][1],
                    alpha=0.1,color=colores_ica[i],label=list(limites_ica.keys())[i])

    plt.ylim(np.min([pd.concat([pm2p5_24h_mean,forecasts]).min().min()+10,0]),\
             np.max([pd.concat([pm2p5_24h_mean,forecasts]).max().max()+10,0]))

    plt.legend(ncol=3,bbox_to_anchor=(1, -0.12),fontsize=13)

    ## TABLE

    # table_subplot = plt.subplot2grid((1, 16), (0, 15))
    # plt.axis('off')
    table = forecasts.iloc[:18]
    table = table.round(2)
    table['date'] = table.index.to_pydatetime().astype(str)
    table['date'] = [table['date'].iloc[i][:-3] for i in range(len(table))]
    if probabilities==False:
        table = table[['date''mean','GB_MO','GB_CH','RF_MO','RF_CH']]
        cell_text = []
        for row in range(len(table)):
            cell_text.append(table.iloc[row])

    #     table.columns = ['Fecha','Promedio [$\mu g / m^3$]','Mínimo [$\mu g / m^3$]','Máximo [$\mu g / m^3$]']
        table.columns = ['Fecha','Promedio [$\mu g / m^3$]','GB-MO [$\mu g / m^3$]',\
                         'GB-CH [$\mu g / m^3$]','RF-MO [$\mu g / m^3$]',\
                         'RF-CH [$\mu g / m^3$]']
    else:
        df_probs = df_probs.round(2)
        table = pd.concat([table,df_probs],axis=1)
        table = table[['date','p25','p75','mean','GB_MO']].dropna()
        cell_text = []
        for row in range(len(table)):
            cell_text.append(table.iloc[row])

    #     table.columns = ['Fecha','Promedio [$\mu g / m^3$]','Mínimo [$\mu g / m^3$]','Máximo [$\mu g / m^3$]']
        table.columns = ['Fecha','Perc. 25 [$\mu g / m^3$]','Perc. 75 [$\mu g / m^3$]','Media [$\mu g / m^3$]',\
                         'GB-MO [$\mu g / m^3$]']
    ptable = plt.table(cellText=cell_text, colLabels=table.columns, bbox=(1.1,0,1,1),edges='vertical',
                      colLoc = 'center',rowLoc='center')
    ptable.auto_set_font_size(False)
    ptable.set_fontsize(12)
    ptable.auto_set_column_width((-1, 0, 1, 2, 3))

    for (row, col), cell in ptable.get_celld().items():
        if (row == 0):
            cell.set_text_props(fontproperties=FontProperties(weight='bold',size=12))

    text = plt.figtext(0.94, 0.91, 'Pronóstico de las próximas 18 horas',fontsize=15)
    t2 = ("* Este producto indica el pronóstico de la concentración de PM2.5 en promedios de 24 horas para\n"
          "las próximas 96 horas en las estaciones poblacionales del Valle de Aburrá.\n\n"
          "* Los modelos se desarrollaron usando información de aerosoles proveniente de CAMS (ECMWF), información satelital\n"
          "de MODIS, e información meteorológica proveniente de GFS (NCEP).\n\n"
          "* Cada línea punteada indica el pronóstico de un método estadístico distinto, y las probabilidades\n"
          "son calculadas a partir de 50 miembros (pronósticos) diferentes del método RF-MO.\n")
    text2 = plt.figtext(0.94, -0.25, t2,fontsize=11, wrap=True)
    
#     plt.show()
    plt.savefig(path_output,bbox_inches='tight')
    plt.close('all')
    
def plot_summary_24h(dic_forecasts_stations, dic_pm_stations,path_output):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    import matplotlib
    import datetime as dt
    
    cmaps = {'Norte':matplotlib.cm.get_cmap('Purples'),
            'Centro':matplotlib.cm.get_cmap('Greens'),
            'Sur':matplotlib.cm.get_cmap('Blues'),
            'Tráfico':matplotlib.cm.get_cmap('Oranges')}

    colores_regiones = {'Norte':'purple',
            'Centro':'green',
            'Sur':'blue',
            'Tráfico':'orange'}


    regiones_estaciones = {'Norte':np.array(['BAR-TORR','COP-CVID','BEL-FEVE','MED-VILL']),        'Centro':np.array(['MED-ARAN','MED-SCRI','MED-SELE','MED-BEME','MED-ALTA','MED-TESO']),        'Sur':np.array(['MED-LAYE','ITA-CJUS','ENV-HOSP','ITA-CONC','EST-HOSP','CAL-JOAR','CAL-LASA','SAB-RAME']),        'Tráfico':np.array(['CEN-TRAF','SUR-TRAF'])}

    limites_ica = {'C.A. Buena':(0,12.5),
                  'C.A. Moderada':(12.5,37.5),
                  'C.A. Dañina a grupos sensibles':(37.5,55.5),
                  'C.A. Dañina a la salud':(55.5,150.5),
                  'C.A. Muy dañina a la salud':(150.5,250.5),
                  'C.A. Peligrosa':(250.5,500)}

    colores_ica = ['green','yellow','orange','red','purple','brown']

    fig = plt.figure(figsize=(11,5))

    tables = {}
    for region in regiones_estaciones.keys():
        tables[region] = pd.DataFrame()

    for i_station,station in enumerate(list(dic_forecasts_stations.keys())):
        if station in regiones_estaciones['Sur']:
            region = 'Sur'
        elif station in regiones_estaciones['Norte']:
            region = 'Norte'
        elif station in regiones_estaciones['Centro']:
            region = 'Centro'
        elif station in regiones_estaciones['Tráfico']:
            region = 'Tráfico'
        
        df_24h = pd.concat([dic_pm_stations[station].rolling(24).mean(),dic_forecasts_stations[station]['MEAN']]).mean(axis=1)[dic_forecasts_stations[station]['MEAN'].index]
        pm2p5_24h = dic_pm_stations[station].rolling(24).mean()

        cmap = cmaps[region]
        where_station = np.where(regiones_estaciones[region]==station)[0][0]
        plt.plot(df_24h, color=cmap((((where_station+1)/(len(regiones_estaciones[region])*2))/0.8)+0.1),\
                 ls = '--',alpha = 0.7)
        plt.plot(pm2p5_24h, color=cmap((((where_station+1)/(len(regiones_estaciones[region])*2))/0.8)+0.1),\
                 alpha = 0.7,label = station)

        tables[region][station] = df_24h
        
    dates_forecast = dic_forecasts_stations[station].index
    forecast_initial_date = dates_forecast[0] - dt.timedelta(hours=1)
    plt.axvline(forecast_initial_date,color='k',ls = '--',label='Fecha inicial de pronóstico')
    ylim = plt.gca().get_ylim()

    #### Background ICA

    for i in range(len(colores_ica)):
        plt.fill_between([dates_forecast[0]-dt.timedelta(hours=48),dates_forecast[-1]],
                    limites_ica[list(limites_ica.keys())[i]][0],
                    limites_ica[list(limites_ica.keys())[i]][1],
                    alpha=0.1,color=colores_ica[i],label=list(limites_ica.keys())[i])

    plt.legend(ncol=3,bbox_to_anchor=(0.98, -0.12),fontsize=13)

    plt.xticks(fontsize=(12))
    plt.yticks(fontsize=(12))
    plt.ylabel('Concentración de PM2.5\n[$\mu g/m^3$]',fontsize=14)
    plt.xlim(dates_forecast[0]-dt.timedelta(hours=12),dates_forecast[-1])
    plt.grid(ls='--',alpha=0.3)
    plt.title('Fecha inicial del pronóstico: '+str(forecast_initial_date)+ ' HL',fontsize=15,loc='left',y = 1.02)

    table = pd.DataFrame()
    for region in regiones_estaciones.keys():
        table[region] = tables[region].mean(axis=1)

        plt.plot(table[region],color=colores_regiones[region],alpha = 1,label = region,lw=2)

    plt.ylim(ylim[0], ylim[1])


    plt.legend(ncol=3,bbox_to_anchor=(0.98, -0.12),fontsize=13)
    table = table.iloc[:18]
    table = table.round(2)
    table['date'] = table.index.to_pydatetime().astype(str)
    table['date'] = [table['date'].iloc[i][:-3] for i in range(len(table))]

    table = table[['date','Norte','Centro','Sur']]
    cell_text = []
    for row in range(len(table)):
        cell_text.append(table.iloc[row])

    table.columns = ['Fecha','Norte [$\mu g / m^3$]','Centro [$\mu g / m^3$]','Sur [$\mu g / m^3$]']
    ptable = plt.table(cellText=cell_text, colLabels=table.columns, bbox=(1,0,1,1),edges='vertical',
                      colLoc = 'center',rowLoc='center')
    ptable.auto_set_font_size(False)
    ptable.set_fontsize(12)
    ptable.auto_set_column_width((-1, 0, 1, 2, 3))

    for (row, col), cell in ptable.get_celld().items():
        if (row == 0):
            cell.set_text_props(fontproperties=FontProperties(weight='bold',size=12))

    text = plt.figtext(0.94, 0.91, 'Pronóstico de las próximas 18 horas en promedio en cada sub-región.',fontsize=15)
    t2 = ("* Este producto indica el pronóstico de la concentración de PM2.5 en promedios de 24 horas para\n"
          "las próximas 96 horas en las estaciones de PM2.5 del Valle de Aburrá.\n\n"
          "* Los modelos se desarrollaron usando información de aerosoles proveniente de CAMS (ECMWF),\n"
          "meteorológica proveniente de GFS (NCEP), y satelital proveniente de MODIS. Cada línea punteada indica\n"
          " el pronóstico promedio (a partir de distintos métodos estadísticos) para cada estación.\n\n")
    text2 = plt.figtext(0.94, -0.25, t2,fontsize=11, wrap=True)

    plt.savefig(path_output,bbox_inches='tight')
    plt.close('all')

def plot_cams_operacional(cams,dates_forecast,path_output,path_output_csv):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    import matplotlib
    import datetime as dt
    
    forecast_initial_date = dates_forecast[0]-dt.timedelta(hours=1)
    
    fig = plt.figure(figsize=(12,12))
    ax1 = fig.add_subplot(211)

    # cams = cams.loc[dates_forecast[-1]dates_forecast[-1]]

    plt.plot(cams.aod,label = 'AOD Total',alpha=1,color='darkblue')
    plt.plot(cams.omaod,label = 'AOD Materia Orgánica',ls='--',alpha=1,color='green')
    plt.plot(cams.suaod,label = u'AOD Sulfato',ls='--',alpha=1,color='gray')
    # plt.plot(cams.bcaod550,label = 'AOD Black Carbon',ls='--',alpha=1,color='black')
    # plt.plot(cams.duaod550,label = 'AOD Dust',ls='--',alpha=1,color='darkorange')
    # plt.plot(cams.ssaod550,label = 'AOD Sea Salt',ls='--',alpha=1,color='darkgray')
    
    plt.axvline(forecast_initial_date,color='k',ls = '--',label='Fecha actual')
    # plt.plot(pm2p5[forecast_initial_date-dt.timedelta(hours=12):],lw=1.4,color='k',label='Observations')

    plt.legend(ncol=1,fontsize=13)

    plt.xticks(fontsize=(12))
    plt.yticks(fontsize=(12))
    plt.ylabel('AOD (550nm)',fontsize=14)

    plt.xlim(dates_forecast[0]-dt.timedelta(hours=12),dates_forecast[-1])
    plt.grid(ls='--',alpha=0.3)

    plt.title('Fecha inicial del pronóstico de CAMS (ECMWF): '+str(cams.index[0])+ ' HL',fontsize=14,loc='left')

    ax2 = fig.add_subplot(212)
    plt.plot(cams.bcaod,label = u'AOD Carbón Negro',ls='--',alpha=1,color='black')
    plt.plot(cams.duaod,label = u'AOD Polvo',ls='--',alpha=1,color='darkorange')
    plt.plot(cams.ssaod,label = u'AOD Sal Marina',ls='--',alpha=1,color='darkgray')

    plt.plot(cams.niaod,label = u'AOD Nitrato',ls='--',alpha=1,color='firebrick')
    plt.plot(cams.amaod,label = u'AOD Amonia',ls='--',alpha=1,color='rebeccapurple')

    plt.axvline(forecast_initial_date,color='k',ls = '--',label='Fecha actual')
    # plt.plot(pm2p5[forecast_initial_date-dt.timedelta(hours=12):],lw=1.4,color='k',label='Observations')

    plt.legend(fontsize=13)

    plt.xticks(fontsize=(12))
    plt.yticks(fontsize=(12))
    plt.ylabel('AOD (550nm)',fontsize=14)

    plt.xlim(dates_forecast[0]-dt.timedelta(hours=12),dates_forecast[-1])
    plt.grid(ls='--',alpha=0.3)
    plt.savefig(path_output,bbox_inches='tight')
    
    cams.to_csv(path_output_csv)
    
def plot_ifrp_forecast(ifrp,dates_forecast,path_output):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    import matplotlib
    import datetime as dt
    
    forecast_initial_date = dates_forecast[0]-dt.timedelta(hours=1)
    
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(111)

    # cams = cams.loc[dates_forecast[-1]dates_forecast[-1]]

    plt.plot(ifrp.rolling(24,center=True,min_periods=6).mean(),\
        label = 'IFRP Pronosticado',alpha=1,color='darkblue')

    plt.axvline(forecast_initial_date,color='k',ls = '--',label='Fecha actual')
    # plt.plot(pm2p5[forecast_initial_date-dt.timedelta(hours=12):],lw=1.4,color='k',label='Observations')

    plt.legend(ncol=1,fontsize=13)

    plt.xticks(fontsize=(12))
    plt.yticks(fontsize=(12))
    plt.ylabel('IFRP [MW]',fontsize=14)

    plt.xlim(ifrp.index[0],dates_forecast[-1])
    plt.grid(ls='--',alpha=0.3)

    plt.title('Potencia radiativa integrada en las retrotrayectorias pronosticadas',fontsize=14,loc='left')
#     plt.show()
    plt.savefig(path_output,bbox_inches='tight')
    
def plot_trajectories_hotspots(path_bt,path_fires,date_forecast,path_output):
    import numpy as np
    import datetime as dt
    import pandas as pd
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
    from cartopy.feature import ShapelyFeature
    import cartopy.feature as cf
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    import shapefile
    import matplotlib.gridspec as gridspec
    import xarray as xr
    import matplotlib.patches as mpatches
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as font_manager
    import copy
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    import sys
    sys.path.insert(1,'/home/jsperezc/jupyter/AQ_Forecast/functions/')
    from trajectories import read_nc
    
    dates_traj,_,lats_traj,lons_traj = read_nc(path_bt)
    df_fires = pd.read_csv(path_fires)
    
    ######### Incendios para los últimos días ############
    str_time = df_fires['acq_time'].values.astype(str)
    str_time = np.array([str_time[i].zfill(4) for i in range(len(str_time))])
    str_date = df_fires['acq_date'].values.astype(str)

    dates_fires = np.array([dt.datetime.strptime(str_date[i]+' '+str_time[i],'%Y-%m-%d %H%M') \
        for i in range(len(str_time))])
    df_fires.index = dates_fires
    df_fires.index = df_fires.index - dt.timedelta(hours = 5)
    
    df_fires = df_fires[str(date_forecast-dt.timedelta(days=4.5)):] 
    lat_fires = df_fires['latitude']
    lon_fires = df_fires['longitude']

    region = [-4,14,-80,-60]
    
    ### FIGURE ####
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111,projection=ccrs.PlateCarree())

    ax.add_feature(cartopy.feature.OCEAN,alpha=0.3)
    ax.add_feature(cartopy.feature.BORDERS,alpha=0.3)
    ax.set_extent([region[2], region[3]-1, region[0], region[1]-1])

    intervalos=21
    mini=1
    maxi=4
    bounds=np.linspace(mini,maxi,intervalos)

    for i in range(len(lats_traj)):
        plt.plot(lons_traj[i],lats_traj[i],color='gray',lw=0.5,alpha=0.5)
    plt.scatter(lon_fires,lat_fires,marker='.',edgecolor='k',s=100,color='red',alpha=0.7)
    

    plt.xlabel('Longitud',fontsize=13)
    plt.ylabel('Latitud',fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    ax.coastlines(alpha=0.5)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    # gl.xlines = False
    # gl.xlocator = mticker.FixedLocator([-180, -45, 0, 45, 180])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 12, 'color': 'k'}
    gl.ylabel_style = {'size': 12, 'color': 'k'}
    # gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
    
    plt.title('Hotspots identificados en los últimos 4 días y retrotrayectorias  \n'+\
        'retrotrayectorias pronosticadas para las próximas 96 horas\n'+\
              str(date_forecast),fontsize=14,loc='left')
    plt.savefig(path_output,bbox_inches='tight')