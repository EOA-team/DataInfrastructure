"""
Created in 10th June 2024

Sélène Ledain
"""
import os
import geopandas as gpd
import xarray as xr
import rioxarray
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import time
import zarr
from datetime import date
from shapely import Polygon
import glob
from shapely.geometry import box


def check_processed_cubes(data_file, datavar, datestart, output_prefix):
    """
    Count how many cubes have been processed for that file and whether file still needs to be processed

    :param data_file: filename
    :param datavar: variable name
    :param datestart: year of data
    :param output_prefix: path where processed data is stored

    :returns process: 1 if file should still be processed/cubes missing, 0 otherwise
    """

    file_pattern = f'MeteoSwiss_{datavar}_*_{int(datestart[:4])}0101_{int(datestart[:4])}1231.zarr' # datavar.split('D')[0]/....
    search_pattern = os.path.join(output_prefix, file_pattern)
    matching_files = glob.glob(search_pattern)
    num_cubes = len(matching_files)

    if 'ch01r' in data_file:
      if num_cubes == 29156:
        process = 1
      else:
        process = 0

    if 'ch01h' in data_file:
      if num_cubes == 29125:
        process = 1
      else:
        process = 0

    return process

def reproj(ds, src_crs, dst_crs):
  """
  Reproject xarray dataset, preserving resolution of the data. Nearest interpolation

  :param ds: xarray dataset
  :param src_crs: source crs
  :param dst_crs: destination crs
  """

  ds = ds.rename({"E": "x", "N": "y"})
  ds = ds.drop_vars(["lat", "lon"])
  res_x = ds.x.values[1] - ds.x.values[0]
  res_y = ds.y.values[1] - ds.y.values[0]
  ds.rio.write_crs("EPSG:2056", inplace=True)
  ds = ds.rio.reproject("EPSG:32632", resolution=(res_x, res_y))

  return ds

def fix_time_coord(ds, datestart):
  """
  Create daily timeseries (numpy datetime64[ns]) given the year. Change the time coordinate to the new timeseries

  :param ds: xarray dataset
  :param datestart: date extracted from filename
  """
  yr = int(datestart[:4])
  n_days = 366 if yr % 4 == 0 and yr % 100 != 0 or yr % 400 == 0 else 365
  time = np.arange(np.datetime64(f'{yr}-01-01'), np.datetime64(f'{yr}-01-01') + np.timedelta64(n_days, 'D'), np.timedelta64(1, 'D'))
  ds = ds.assign_coords(time=time)

  return ds

def regrid_product_cube(product_cube, lon_lat_grid):
  """
  Regrid xarray to a lon lat grid using nearest interpolation

  :param product_cube: xarray dataset
  :param lon_lat_grid: tuple of lon and lat grid
  """

  # Store original data types
  original_dtypes = {var: product_cube[var].dtype for var in product_cube.variables}

  if ("x" in product_cube.coords) and ("y" in product_cube.coords):

      x, y = product_cube.x.values, product_cube.y.values
      lon_grid, lat_grid = lon_lat_grid
      
      product_cube = product_cube.interp(x = lon_grid, y = lat_grid, method = "nearest")

      product_cube = product_cube.rename({"x": "lon", "y": "lat"})

  return product_cube

def slice_and_save(ds, grid, datavar, datestart, output_prefix, compressor, overwrite):
  """
  Regrid the weather data to the grid and save the datacubes to zarr

  :param ds: xarray dataset
  :param grid: geopandas dataframe
  :param datavar: variable name
  :param datestart: year of data
  :param output_prefix: path to save the zarr files
  :param compressor: zarr compressor
  :param overwrite: boolean to overwrite existing files
  """

  for i, row in grid.iterrows(): 
      print(f'Processing grid patch {i}/{len(grid)}')
      # If files are stored in data var subfolders, could implement an easier check of whether the cubes already processed

      patch = row.geometry
      minx, miny, maxx, maxy = patch.bounds

      """
      output_path = output_prefix + f'MeteoSwiss_{datavar}_{int(minx)}_{int(maxy)}_{int(datestart[:4])}0101_{int(datestart[:4])}1231.zarr'
      #output_path = output_prefix + f'{data_var.split('D')[0]}/MeteoSwiss_{datavar}_{int(minx)}_{int(maxy)}_{int(datestart[:4])}0101_{int(datestart[:4])}1231.zarr'
      matching_cubes = glob.glob(output_path)
      if not len(matching_cubes):
      """

      lon_lat_grid = [np.arange(minx, maxx, 10), np.arange(miny+10, maxy+10, 10)] # make sure that last upper left corner is produced
      regrid = regrid_product_cube(ds, lon_lat_grid) 

      if not np.isnan(regrid[datavar]).all():

        # Drop addition variables
        regrid = regrid.drop_vars(["swiss_lv95_coordinates"], errors="ignore")

        # Update metadata
        attrs = regrid.attrs
        attrs['history'] += f". Reprojected and regrid datacube to EPSG 32632 by Sélène Ledain on {date.today()}"
        regrid.attrs = attrs
        
        # Chunk
        regrid = regrid.chunk({'time': -1, 'lat': -1, 'lon': len(regrid.lon)/2}) 

        # Save the data to zarr meteo_var_minx_maxy_datestart_datend.zarr
        output_path = output_prefix + f'{data_var.split('D')[0]}/MeteoSwiss_{datavar}_{int(minx)}_{int(maxy)}_{int(datestart[:4])}0101_{int(datestart[:4])}1231.zarr'

        # Save the patch to Zarr with compression
        if overwrite or not os.path.exists(output_path):
            regrid.to_zarr(output_path, consolidated=True, mode='w', encoding={var: {'compressor': compressor} for var in regrid.data_vars})
            print('Saved patch', output_path) # save_end-save_start

  return




if __name__ == "__main__":

  print('STARTING METEO FILES PROCESSING')
    
  #####################
  # DEFINE PATHS AND VARIABLES

  grid_path = '~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/gridface_s2tiles_CH.shp'
  grid = gpd.read_file(grid_path)

  output_prefix = os.path.expanduser('~/mnt/eo-nas1/data/meteo/')
  overwrite = False # If True, will overwrite existing files of same name

  data_path = os.path.expanduser('~/mnt/Data-Raw-RE/27_Natural_Resources-RE/99_Meteo_Public/MeteoSwiss_netCDF/__griddedData/lv95updated')
  data_files = [f for f in os.listdir(data_path) if f.endswith('.nc') and not f.startswith('topo') and f.split('_')[0].endswith('D')]

  compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)

 
  #####################
  # PROCESS FILES

  #processed_files = [f for f in os.listdir(output_prefix)]
  
  for i, data_file in enumerate(data_files):
    datavar, _, datestart, _ = data_file.split("_") #varRes_gridtype_date.nc

    # Check if file should be processed
    #process = check_processed_cubes(data_file, datavar, datestart, output_prefix)        
    if 'ch01r' in data_file: #process:
      print(f'-------Processing file {i}/{len(data_files)}: {data_file}-----------')

      ds = xr.open_dataset(os.path.join(data_path, data_file), decode_times=False) 
      # Fix time coordinate: get year and create monthly or daily or yearly timeseries
      ds = fix_time_coord(ds, datestart)
      # Reproject file to EPSG 32632
      ds = reproj(ds, 2056, 32632)
      # Regrid and save the data
      slice_and_save(ds, grid, datavar, datestart, output_prefix, compressor, overwrite)

              
