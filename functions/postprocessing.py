#Funcion que permite hacer recorte espacial de archivos NetCDF provenientes de GEOS5
import numpy as np
import xarray as xr
from scipy import spatial
def recorte_datos_GEOS5(path_files, coordenadas):
    """
    Esta funcion Permite recortar archivos NetCDF del conjunto de datos de GEOS5 haciendo uso de la libreria xarray.
    
    Para esto, debe ingresar la ruta del directorio donde estan los archivos a recortar, ademas de un vector con las 
    coordenadas 
    en las siguientes posiciones: [N,S,E,O] 
    
    Tenga en cuenta que se guardara el nuevo archivo con el mismo nombre que el archivo original, es decir, se va a 
    sobreescribir 
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
        print(file)
        #Hacemos el recorte de los datos a nuestra region de interés
        split_data = global_data.loc[dict(lon=lon[(lon > coordenadas[3]) & (lon < coordenadas[2])], 
                                          lat=lat[(lat > coordenadas[1]) & (lat < coordenadas[0])])] 

        os.system('rm {}'.format(file)) 
        #Guardamos el nuevo NetCDf
        split_data.to_netcdf(path_files+Name_Files[i], 
                                  unlimited_dims='time',format="NETCDF4_CLASSIC") 

#Funcion para recortar datos de cualquier conjunto de datos -> A diferencia de la funcion anterior, esta no genera un nuevo
#NetCDF.
def recorte_espacial(Dataset):   
    
    """
    Esta funcion permite hacer un recorte espacial de la malla original de cualquier conjuntos de datos, a solo tener el 
    Valle de Aburra. Atributos de entrada:
    1. Dirección del archivo NetCDF
    2. Y el vector de coordenadas así [O,S,E,N]
    -> Se recomienda utilziar este vector para el Valle de Aburra: [-75.71943, 5.978178, -75.222057, 6.512488]
    """
    import xarray as xr
    #coordenadas del Valle
    coords = [-75.85, 5.978178, -75.222057, 6.512488]
    
    #Algunos archivos traen el nombre de coordenadas como: lon y lat y otros como: longitude y latitude
    try:
        lon = Dataset.coords["longitude"]
        lat = Dataset.coords["latitude"]
        #Hacemos el recorte de los datos a nuestra region de interés
        Dataset = Dataset.loc[dict(longitude=lon[(lon >= coords[0]) & (lon <= coords[2])], 
                                          latitude=lat[(lat >= coords[1]) & (lat <= coords[3])])] 
    except:
        lon = Dataset.coords["lon"]
        lat = Dataset.coords["lat"]
        #Hacemos el recorte de los datos a nuestra region de interés
        Dataset = Dataset.loc[dict(lon=lon[(lon > coords[0]) & (lon < coords[2])], 
                                          lat=lat[(lat > coords[1]) & (lat < coords[3])])] 

    return Dataset

def call_files(conjunto):
    import xarray as xr
    import os
    import glob
    """
    conjunto = 'CAMS' o conjunto = 'GEOS5'
    
    Esta funcion permite cargar todos los archivos NetCDF (de todos los años) de CAMS y/o de GEOS5 en 
    un solo archivo
    Nota: Las variables de GEOS5 que se cargaran son: 'BCSCATAU','DUSCATAU','OCSCATAU','SSSCATAU','TOTSCATAU'
    Nota2: Las variables pm2.5 y pm10 se convierten a micogramos[ug]
    """
    #En esta sesion alamcenaremos la ruta de cada archivo de CAMS en un vector 
    if conjunto == 'CAMS':
        path = '/var/data1/AQ_Forecast_DATA/historic/CAMS/Reanalisis/*'
        path_files = []
        for i in (sorted(glob.glob(path))):
            for j in (sorted(glob.glob(i+'/*'))):
                for k in (sorted(glob.glob(j+'/*'))):
                    path_files.append(k)
        #Cargamos TODOS los archivos que tenemos de CAMS en un solo Dataset
        data=xr.open_mfdataset(path_files, autoclose='True') #Hacemos el merge con xarray
        #Convertimos kg a ug
        data['pm2p5'] = data.pm2p5*1000_000_000
        data['pm10'] = data.pm10*1000_000_000 
        
    elif conjunto == 'GEOS5':
        path = '/var/data1/AQ_Forecast_DATA/historic/GEOS5/Reanalisis/*'
        path_files = []
        for i in (sorted(glob.glob(path))):
            for j in (sorted(glob.glob(i+'/*'))):
                for k in (sorted(glob.glob(j+'/*'))):
                    path_files.append(k)
        #Vector de variables innecesarias
        variables_GEOS = ['BCSMASS', 'DMSSMASS', 'DUSMASS25', 'OCSMASS', 'SO2SMASS', 'SSSMASS25',
                          'HNO3SMASS', 'NH3SMASS', 'NISMASS25', 'SO4SMASS',
                          'BCEXTTAU', 'DUEXTTAU', 'OCEXTTAU',
                          'SSEXTTAU', 'SUEXTTAU','TOTEXTTAU']
        #Cargamos TODOS los archivos que tenemos de CAMS en un solo Dataset
        data=xr.open_mfdataset(path_files[:], concat_dim='time', autoclose='True')[variables_GEOS]#Merge con xarray
    return data
    
    
    
    
def find_pixel(latitud, longitud, dataset):
    """
    función para determinar a qué pixel de un archivo netcdf pertenece una ubicación cualquiera
    inputs:
    latitud, longitud: latitud y longitud del punto
    dataset: puede ser una ruta donde se alojan los archivos o el set de datos ya abierto
    """
    
    if isinstance(dataset, str):
        dataset = xr.open_mfdataset(dataset)
    else:
        longitudes, latitudes = np.array(longitud), np.array(latitud)
        try:
            xs, ys = np.meshgrid(dataset.longitude, dataset.latitude)
        except:
            xs, ys = np.meshgrid(dataset.lon, dataset.lat)
        arbol_binario = spatial.cKDTree(np.c_[xs.ravel(), ys.ravel()])
        
        lxs, lys = np.meshgrid(longitudes, latitudes)
        distancias, indices = arbol_binario.query(np.c_[lxs.ravel(), lys.ravel()])
        
        Dataset = dataset.loc[dict(longitude=[np.c_[xs.ravel(), ys.ravel()][indices][0][0]], 
                                   latitude=[np.c_[xs.ravel(), ys.ravel()][indices][0][1]])]
    return Dataset
