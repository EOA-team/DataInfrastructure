""" 
Quickly view the data in a zarr store
Will show RGB image of cube for the least cloudy date

Sélène Ledain
Aug 15th 2024
"""

from argparse import ArgumentParser
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
import pandas as pd
import rioxarray



def view(cube_path):
  """ 
  Plot RGB image of cube on least cloudy date

  :param cube_path: str, patht to zarr sotre
  """
  dataset = xr.open_zarr(cube_path).compute()
  cloud_coverage = (dataset['s2_mask'] != 0).mean(dim=['lat', 'lon'])
  least_cloudy_date = cloud_coverage.argmin()
  rgb = dataset.isel(time=least_cloudy_date)[['s2_B04', 's2_B03', 's2_B02']].to_dataarray('band')

  # Normalize the data for better visualization (optional)
  rgb = rgb.where(rgb != 65535, np.nan)
  rgb = rgb.where(~np.isnan(rgb), other=1.0)
  rgb_normalized = rgb / rgb.max()
  # Plot the RGB image
  plt.figure(figsize=(10, 10))
  plt.imshow(rgb_normalized.transpose('lat', 'lon', 'band'))
  plt.title(f"Least Cloudy date for {cube_path.split('/')[-1].split('.zarr')[0]}")
  plt.axis('off')
  plt.savefig(f"quick_view_{cube_path.split('/')[-1].split('.zarr')[0]}.png")
  plt.show()

  return 


def open_cubes_conflicting(cubes):
  """
  Open and combine zarr files where there are conflicts along the itme dimensions due to multiple tiles/multiple acquisitions at a same timestamps. 
  Use product_uri to merge along time correctly

  :param cubes: list of zarr files
  :returns ds: combined zarr files in xr Dataset
  """
  grouped_datasets = defaultdict(list)

  # Load each Zarr file and organize it by (timestamp, scene_id)
  for zarr_path in cubes:
      ds = xr.open_dataset(zarr_path, engine="zarr").compute()
      for i in range(ds.sizes['time']):
          ds_time_slice = ds.isel(time=i)
          timestamp = ds_time_slice['time'].values
          scene_id = ds_time_slice.scene_id.item()  # Assuming scene_id is scalar for each time slice
          
          # Use a combination of timestamp, scene_id, as the key
          key = (pd.Timestamp(timestamp), scene_id)
          grouped_datasets[key].append(ds_time_slice)
  
  # Combine datasets with matching (timestamp, tile_id, granule_id)
  combined_datasets = []
  for i, ((timestamp, scene_id), datasets) in enumerate(grouped_datasets.items()):
      if len(datasets) > 1:
          #datasets[5][['s2_B04', 's2_B03', 's2_B02']].rename({'lat':'y', 'lon':'x'}).rio.to_raster(f'ds5_{tile_id}_{granule_id}.tif')
          combined_ds = xr.combine_by_coords(datasets, combine_attrs='override')
          #combined_ds[['s2_B04', 's2_B03', 's2_B02']].rename({'lat':'y', 'lon':'x'}).rio.to_raster(f'{tile_id}_{granule_id}.tif')
          #combined_ds[['mean_sensor_azimuth', 'mean_sensor_zenith','mean_solar_azimuth', 'mean_solar_zenith', 'product_uri']] = \
          #  datasets[0][['mean_sensor_azimuth', 'mean_sensor_zenith','mean_solar_azimuth', 'mean_solar_zenith', 'product_uri']]
      else:
        combined_ds = datasets[0]
      
      combined_datasets.append(combined_ds)
  
  ds = xr.concat(combined_datasets, dim="time")

  return ds


if __name__ ==  "__main__":

  landsat_data = os.path.expanduser('~/mnt/eo-nas1/data/satellite/landsat/raw/CH/89/')
  x, y = (300630, 5179260)
  side = 3
  side_len = 128*30 #meters
  year = 2021

  # Find data at (x, y) of specified sizes
  landsat_files = [os.path.join(landsat_data, f) for f in os.listdir(landsat_data) if f.endswith('.zarr')]
  cubes = []
  for minx in range(x, x + side*side_len, side_len):
    for maxy in range(y, y - side*side_len, -side_len):
      f = [f for f in landsat_files if f'LS_{minx}_{maxy}_{year}' in f][0]
      cubes.append(f)
  
  #ds = open_cubes_conflicting(cubes)
  #print(ds.time.values[50])
  #ds.isel(time=50).OLI_B4.rename({'lat':'y', 'lon': 'x'}).rio.set_crs(32632).rio.to_raster('cubes.tif')

  for cube in cubes:
    ds = xr.open_zarr(cube).compute()
    print(ds.time.values[50])
    ds.isel(time=50).OLI_B4.rename({'lat':'y', 'lon': 'x'}).rio.set_crs(32632).rio.to_raster('cube0.tif')
    #print(ds.sel(time='2021-08-14').OLI_B4)
    break

  