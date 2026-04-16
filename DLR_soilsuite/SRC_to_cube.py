""" 
Convert raw data from DLR soilsuite into cube format

Raw data:
EPSG:3035, grid and tile format provided by DLR. There are 4 tiles covering CH: 0040-0024, 0040-0026, 0042-0024, 0042-0026
The resolution is 20m and coordinates are the center of the pixel

Sélène Ledain
9th May 2025
"""

import xarray as xr
import os
import geopandas as gpd
from pyproj import Transformer
import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def round_down_to_nearest_10(value):
    return value - (value % 10)

def round_up_to_nearest_10(value):
    return math.ceil(value / 10) * 10


raw_data_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/raw/DLR_soilsuite/')
out_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/DLR_soilsuite/')
s2_grid = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/gridface_s2tiles_CH.shp')
grid_shp = gpd.read_file(s2_grid, crs=36232)


for n_file, f in enumerate(os.listdir(raw_data_dir)):
  ds = xr.open_zarr(os.path.join(raw_data_dir, f)).compute().drop_vars('spatial_ref')
 
  # To reproject to S2 grid in EPSG:32632 :
  # - Adjust coordinates to become top left of pixel (instead of center)
  # - Get bounds in EPSG:3035
  # - Transform to EPSG:32632
  # - Find nearest 10m
  # - Create grid in EPSG:32632
  # - Transform to EPSG:3035
  # - Interpolate data (make sure to identify fill/missing values first, to not interpolate those)
  # - Update coordinates to those in EPSG:32632
  ds = ds.assign_coords({
    'x': ds.x - 10,
    'y': ds.y + 10
  })
  minx, miny, maxx, maxy = ds.x.values.min(), ds.y.values.min(), ds.x.values.max(), ds.y.values.max()
  #print(minx, miny, maxx, maxy)

  transformer_to_utm = Transformer.from_crs(3035, 32632, always_xy=True)
  minx_utm32, miny_utm32, maxx_utm32, maxy_utm32 = transformer_to_utm.transform_bounds(minx, miny, maxx, maxy)
  minx_utm32 = round_down_to_nearest_10(minx_utm32)
  miny_utm32 = round_down_to_nearest_10(miny_utm32)
  maxx_utm32 = round_up_to_nearest_10(maxx_utm32)
  maxy_utm32 = round_up_to_nearest_10(maxy_utm32)

  x_s2 = np.arange(minx_utm32, maxx_utm32 + 10, 10)
  y_s2 = np.arange(maxy_utm32, miny_utm32 - 10, -10)
  #print(len(x_s2), len(y_s2))
  X_s2, Y_s2 = np.meshgrid(x_s2, y_s2, indexing='ij') 

  transformer_to_3035 = Transformer.from_crs(32632, 3035, always_xy=True)
  x_s2_3035, y_s2_3035 = transformer_to_3035.transform(X_s2, Y_s2)
  #print(x_s2_3035.shape, y_s2_3035.shape)
  #print(x_s2_3035[0][0], y_s2_3035[0][0], x_s2_3035[-1][0], y_s2_3035[0][-1])

  # Replace missing values with NaN
  fill_is_neg10 = [var for var in ds.data_vars if 'CI95' in var or 'STD' in var]
  fill_is_neg10k = [var for var in ds.data_vars if var.startswith('SRC_B')]
  fill_is_0 = ['MASK']
  for v in fill_is_neg10:
    ds[v] = ds[v].where(ds[v] != -10, np.nan)
  for v in fill_is_neg10k:
    ds[v] = ds[v].where(ds[v] != -10000, np.nan)
  for v in fill_is_0:
    ds[v] = ds[v].where(ds[v] != 0, np.nan)
  
  # Interpolate data to S2 grid
  old_arr = ds.to_array().values # shape (vars, y, x)
  reproj_arr = np.zeros((old_arr.shape[0], x_s2_3035.shape[1], x_s2_3035.shape[0])) # shape (vars, y, x)
  for v in range(old_arr.shape[0]):
    f = RegularGridInterpolator((ds.y.values, ds.x.values), old_arr[v, :, :], method='linear',bounds_error=False, fill_value=np.nan)
    reproj_arr[v, :, :] = f((y_s2_3035, x_s2_3035)).T.astype(np.float64)


  # Create new dataset
  data_vars = {}
  original_vars = list(ds.data_vars)

  for i, var_name in enumerate(original_vars):
      data_vars[var_name] = xr.DataArray(
          reproj_arr[i, :, :],
          coords={'y': y_s2, 'x': x_s2},
          dims=('y', 'x'),
          attrs=ds[var_name].attrs  # preserve attributes
      )
  ds_reproj = xr.Dataset(data_vars)
  ds_reproj.attrs = ds.attrs.copy()
  ds_reproj.attrs.update({
      'reprocessing': 'Reprojected to EPSG:32632 and interpolated to S2 grid by Selene Ledain, 05.2025'
  })


  # Chunk up to S2 datacubes
  for i, cube in grid_shp.iterrows():
    left = cube.left
    top = cube.top

    save_path = os.path.join(out_dir, f'SRC_{int(left)}_{int(top)}.zarr')
    if not os.path.exists(save_path):

      if left in ds_reproj.x.values and top in ds_reproj.y.values and left+1270 in ds_reproj.x.values and top+1270 in ds_reproj.y.values:
        ds_chunk = ds_reproj.sel(x=slice(left, left + 1270), y=slice(top, top -1270))

        if len(ds_chunk.x.values) != 128 or len(ds_chunk.y.values) != 128:
          # In case the coords were out of bounds, will make sure that it is filled with Nan
          ds_chunk = ds_reproj.reindex(x=np.arange(left, left+1280, 10), y=np.arange(top, top-1280, -10), method=None)
          
        
        if not np.isnan(ds_chunk.to_array().values).all():
          for var_name in original_vars:
            # Replace nan values back with fill values, and convert to int16
            if var_name in fill_is_neg10:
              ds_chunk[var_name] = xr.where(ds_chunk[var_name].isnull(), -10, ds_chunk[var_name])
            elif var_name in fill_is_neg10k:
              ds_chunk[var_name] = ds_chunk[var_name] = xr.where(ds_chunk[var_name].isnull(), -10000, ds_chunk[var_name])
            elif var_name in fill_is_0:
              ds_chunk[var_name] = xr.where(ds_chunk[var_name].isnull(), 0, ds_chunk[var_name])
            ds_chunk[var_name] = ds_chunk[var_name].round().astype('int16')

          # Save
          save_path = os.path.join(out_dir, f'SRC_{int(left)}_{int(top)}.zarr')
          ds_chunk.to_zarr(save_path, mode='w', consolidated=True)
          print(f'Saved cube {i}/{len(grid_shp)}: {save_path}')
