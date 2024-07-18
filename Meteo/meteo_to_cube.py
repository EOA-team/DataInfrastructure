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



def slice_and_save(ds, grid, datavar, output_prefix, compressor, overwrite):
  """
  Regrid the weather data to the grid and save the datacubes to zarr

  :param ds: xarray dataset
  :param grid: geopandas dataframe
  :param datavar: variable name
  :param output_prefix: path to save the zarr files
  :param compressor: zarr compressor
  :param overwrite: boolean to overwrite existing files
  """

  for i, row in grid.iterrows(): 
        print(f'Processing grid patch {i}/{len(grid)}')
        patch = row.geometry
        minx, miny, maxx, maxy = patch.bounds
        lon_lat_grid = [np.arange(minx, maxx, 10), np.arange(miny, maxy, 10)] # make sure that last upper left corner is produced

        regrid = regrid_product_cube(ds, lon_lat_grid) 
        if not np.isnan(regrid[datavar]).all():

          # Drop addition variables
          regrid = regrid.drop_vars(["swiss_lv95_coordinates"], errors="ignore")

          # Update metadata
          attrs = regrid.attrs
          attrs['history'] += f". Reprojected and regrid datacube to EPSG 32632 by Sélène Ledain on {time.time()}"
          regrid.attrs = attrs
          
          # Chunk
          regrid = regrid.chunk({'time': -1, 'lat': -1, 'lon': len(regrid.lon)/2}) 

          # Save the data to zarr meteo_var_minx_maxy_datestart_datend.zarr
          output_path = output_prefix + f'MeteoSwiss_{datavar}_{int(minx)}_{int(maxy)}_{int(datestart[:4])}0101_{int(datestart[:4])}1231.zarr'

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
  data_files = [f for f in os.listdir(data_path) if f.endswith('.nc') and not f.startswith('topo')]

  compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)


  #####################
  # PROCESS FILES

  for i, data_file in enumerate(data_files):
    print(f'-------Processing file {i}/{len(data_files)}-----------')

    datavar, _, datestart, _ = data_file.split("_") #varRes_gridtype_date.nc

    # Only process daily data
    if datavar.endswith('D'):
      ds = xr.open_dataset(os.path.join(data_path, data_file), decode_times=False) 

      # Fix time coordinate: get year and create monthly or daily or yearly timeseries
      ds = fix_time_coord(ds, datestart)

      # Reproject file to EPSG 32632
      ds = reproj(ds, 2056, 32632)

      # Regrid and save the data
      slice_and_save(ds, grid, datavar, output_prefix, compressor, overwrite)
            
