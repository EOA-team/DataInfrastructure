""" 
Coregistration pipeline using SwissImage to correct S2 data
"""

import dask
from distributed import Client
import os
import xarray as xr
import numpy as np
import geopandas as gpd
import glob
import warnings
warnings.filterwarnings('ignore')
from geoarray import GeoArray
from arosics import COREG
from arosics import DESHIFTER
from pyproj import CRS
from shapely import Polygon, box
import zarr
import datetime
import pandas as pd
from collections import defaultdict
import rioxarray
from PIL import Image
from scipy.ndimage import affine_transform
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextily as cx
import json
import pickle


def extract_ids(product_uri):
  """
  Extract tile_id and granule_id from product_uri string
  """
  parts = product_uri.split('_')
  tile_id = parts[5]  # e.g., T31TGM
  granule_id = parts[-1].split('.')[0]  # e.g., 20210529T141943
  return tile_id, granule_id


def open_cubes_conflicting(cubes):
  """
  Open and combine zarr files where there are conflicts along the itme dimensions due to multiple tiles/multiple acquisitions at a same timestamps. 
  Use product_uri to merge along time correctly

  :param cubes: list of zarr files
  :returns ds: combined zarr files in xr Dataset
  """
  grouped_datasets = defaultdict(list)

  # Load each Zarr file and organize it by (timestamp, tile_id, granule_id)
  for zarr_path in cubes:
      ds = xr.open_dataset(zarr_path, engine="zarr").compute()
      for i in range(ds.dims['time']):
          ds_time_slice = ds.isel(time=i)
          timestamp = ds_time_slice['time'].values
          product_uri = ds_time_slice.product_uri.item()  # Assuming product_uri is scalar for each time slice
          tile_id, granule_id = extract_ids(product_uri)
          
          # Use a combination of timestamp, tile_id, and granule_id as the key
          key = (pd.Timestamp(timestamp), tile_id, granule_id)
          grouped_datasets[key].append(ds_time_slice)
  
  # Combine datasets with matching (timestamp, tile_id, granule_id)
  combined_datasets = []
  for i, ((timestamp, tile_id, granule_id), datasets) in enumerate(grouped_datasets.items()):
      if len(datasets) > 1:
          #datasets[5][['s2_B04', 's2_B03', 's2_B02']].rename({'lat':'y', 'lon':'x'}).rio.to_raster(f'ds5_{tile_id}_{granule_id}.tif')
          combined_ds = xr.combine_by_coords(datasets, combine_attrs='override')
          #combined_ds[['s2_B04', 's2_B03', 's2_B02']].rename({'lat':'y', 'lon':'x'}).rio.to_raster(f'{tile_id}_{granule_id}.tif')
          combined_ds[['mean_sensor_azimuth', 'mean_sensor_zenith','mean_solar_azimuth', 'mean_solar_zenith', 'product_uri']] = \
            datasets[0][['mean_sensor_azimuth', 'mean_sensor_zenith','mean_solar_azimuth', 'mean_solar_zenith', 'product_uri']]
      else:
        combined_ds = datasets[0]
      
      combined_datasets.append(combined_ds)
  
  ds = xr.concat(combined_datasets, dim="time")

  return ds


def extract_bounds_multiyear(file):
    """
    Extract different parts of the S2+LAI file names

    :param file: str
    """
    parts = file.split('_')
    minx = int(parts[1])
    maxx = minx + 1280
    maxy = int(parts[2])
    miny = maxy - 1280
    yr_start = int(parts[3][:4])
    yr_end = int(parts[4][:4])
    return minx, miny, maxx, maxy, yr_start, yr_end

    
def find_cubes_aoi_yrs(data_folder, geom, yrs):
  """
  Find S2 cubes that fall in perimeter given by geometry

  :param data_folder: path to where S2 data is stored
  :param geom: gpd.GeoDataFrame containing geometry to use to filter files
  :param yrs: list of integer years of interest
  :return: gdf with filtered files as rows
  """
  # 1. Filter S2 based on geom intersection
  cubes = [f for f in os.listdir(data_folder) if f.endswith('zarr')]
  df_cubes = pd.DataFrame(cubes, columns=['file'])
  df_cubes[['minx', 'miny', 'maxx', 'maxy', 'yr_start', 'yr_end']] = df_cubes['file'].apply(lambda x: pd.Series(extract_bounds_multiyear(x)))
  df_cubes['geometry'] = df_cubes.apply(lambda row: box(row['minx'], row['miny'], row['maxx'], row['maxy']), axis=1)
  gdf_cubes = gpd.GeoDataFrame(df_cubes, geometry='geometry')
  filtered_files = gdf_cubes[gdf_cubes.intersects(geom.unary_union)]
  filtered_files = filtered_files[(filtered_files['yr_start'].isin(yrs)) | (filtered_files['yr_end'].isin(yrs))]
  
  return filtered_files


def load_cubes(f, target_folder):
  """
  Find contiguous cubes to the on in f. Load all of the cubes to an xarray dataset

  :param f: path to cube
  :param target_folder: path to folder where f is stored
  :returns ds: xr.Dataset with all cubes
  :returns minx, maxy: topleft coordinate of central cube
  :returns central_cube_attrs: metadata of central cube
  """
   
  # Get info on file coordinates
  minx, maxy = int(f.split('/')[-1].split('_')[1]), int(f.split('/')[-1].split('_')[2])
  base_dir = f.split('/')[:-1]
  # Calculate contiguous cubes
  xs = np.arange(minx-1280, minx+1280*2, 1280)
  ys = np.arange(maxy-1280, maxy+1280*2, 1280)

  # Find files that have these coords (all years)
  file_patterns = [os.path.join(target_folder, f'S2_{x}_{y}_*.zarr') for x in xs for y in ys]
  cubes = [file for pattern in file_patterns for file in glob.glob(pattern)]
  #cubes = ['/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5113260_20170103_20171231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5113260_20210102_20211230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5113260_20230102_20231230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5113260_20180103_20181231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5113260_20220102_20221230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5113260_20190103_20191231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5113260_20200103_20201230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5114540_20170103_20171231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5114540_20210102_20211230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5114540_20230102_20231230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5114540_20180103_20181231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5114540_20220102_20221230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5114540_20190103_20191231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5114540_20200103_20201230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5115820_20170103_20171231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5115820_20210102_20211230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5115820_20230102_20231230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5115820_20190103_20191231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5115820_20180103_20181231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5115820_20220102_20221230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5115820_20200103_20201230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5113260_20170103_20171231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5113260_20210102_20211230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5113260_20230102_20231230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5113260_20180103_20181231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5113260_20220102_20221230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5113260_20190103_20191231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5113260_20200103_20201230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5114540_20170103_20171231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5114540_20210102_20211230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5114540_20230102_20231230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5114540_20180103_20181231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5114540_20220102_20221230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5114540_20190103_20191231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5114540_20200103_20201230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5115820_20170103_20171231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5115820_20210102_20211230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5115820_20220102_20221230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5115820_20230102_20231230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5115820_20180103_20181231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5115820_20190103_20191231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5115820_20200103_20201230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5113260_20170103_20171231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5113260_20210102_20211230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5113260_20230102_20231230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5113260_20180103_20181231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5113260_20220102_20221230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5113260_20190103_20191231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5113260_20200103_20201230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5114540_20170103_20171231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5114540_20210102_20211230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5114540_20230102_20231230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5114540_20180103_20181231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5114540_20220102_20221230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5114540_20190103_20191231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5114540_20200103_20201230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5115820_20170103_20171231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5115820_20210102_20211230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5115820_20220102_20221230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5115820_20230102_20231230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5115820_20190103_20191231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5115820_20180103_20181231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5115820_20200103_20201230.zarr']
  
  try:
    # No conflicting timestamps
    ds = xr.open_mfdataset(cubes, combine='by_coords').compute()
  except:
    ds = open_cubes_conflicting(cubes)

  central_cube_attrs = [xr.open_zarr(f).attrs for f in cubes if f'{minx}_{maxy}' in f][0]
  central_cube = [xr.open_zarr(f) for f in cubes if f'{minx}_{maxy}' in f][0]
 
  return ds, minx, maxy, central_cube_attrs


def has_all_65535(ds):
    return ((ds == 65535).all(dim=['lat', 'lon'])).to_array().sum()


def has_clouds(ds, cloud_thresh=0.1):
    cloud_condition = (ds.s2_mask == 1) & (ds.s2_SCL.isin([8, 9, 10]))
    return cloud_condition.sum(dim=['lat', 'lon'])/(len(ds.lat)*len(ds.lon)) > cloud_thresh


def has_shadows(ds, shadow_thresh=0.1):
    shadow_condition = (ds.s2_mask == 2) & (ds.s2_SCL == 3)
    return shadow_condition.sum(dim=['lat', 'lon'])/(len(ds.lat)*len(ds.lon)) > shadow_thresh


def has_snow(ds, snow_thresh=0.1):
    snow_condition = (ds.s2_mask == 3) & (ds.s2_SCL == 11)
    return snow_condition.sum(dim=['lat', 'lon'])/(len(ds.lat)*len(ds.lon)) > snow_thresh


def has_cirrus(ds, cirrus_thresh=1000):
    cirrus_mask = ds.s2_SCL == 10
    cirrus_b02_mean = ds.s2_B02.where(cirrus_mask).mean(dim=['lat', 'lon'])
    return cirrus_b02_mean > cirrus_thresh


def clean_dataset(ds, cloud_thresh=0.1, shadow_thresh=0.1, snow_thresh=0.1, cirrus_thresh=1000):
  """
  Drop dates with no data and clouds/snow/shadows
  """
  n_times = len(ds.time)

  # Remove cloudy or missing dates: any of the bands is all 65535
  dates_to_drop = [i for i, date in enumerate(ds.time.values) if has_all_65535(ds.isel(time=i))]
  mask_dates = np.ones(len(ds.time), dtype=bool)
  mask_dates[dates_to_drop] = False
  ds = ds.isel(time=mask_dates)

  # Remove too many clouds (mask=1), shadows (mask=2) or snow (mask=3)
  dates_to_drop = [i for i, date in enumerate(ds.time.values) if has_clouds(ds.isel(time=i), cloud_thresh)] + \
                [i for i, date in enumerate(ds.time.values) if has_shadows(ds.isel(time=i), shadow_thresh)] + \
                [i for i, date in enumerate(ds.time.values) if has_snow(ds.isel(time=i), snow_thresh)] +\
                [i for i, date in enumerate(ds.time.values) if has_cirrus(ds.isel(time=i), cirrus_thresh)]
  mask_dates = np.ones(len(ds.time), dtype=bool)
  mask_dates[dates_to_drop] = False
  ds = ds.isel(time=mask_dates)

  return ds


def apply_shifts(image, shifts, output_shape):
    """
    Apply affine transform on image given shifts in y and x dimension. Since the image dimensions are (y,x,band), shifts should be (shift_y, shift_x)
    """
    # Function to apply affine transformation based on shifts
    matrix = np.array([[1, 0, shifts[1]], [0, 1, shifts[0]], [0, 0, 1]])
    transformed_image = np.zeros(output_shape, dtype=image.dtype)
    for band in range(image.shape[2]):
        transformed_image[:, :, band] = affine_transform(image[:, :, band], matrix, mode='nearest', output_shape=output_shape[:2])
    
    return transformed_image


# Parallelize each coregistration step with Dask
@dask.delayed
def coreg_single_step(i, ds_tgt, geo_ref_image, geotransform_tgt, projection_tgt, footprint_tgt, geotransform_ref, projection_ref, footprint_ref, coreg_mask):
  
    # Same coreg logic for one timestep
    target_image = ds_tgt.isel(time=i).s2_B04
    geo_tgt_image = GeoArray(target_image.values, geotransform=geotransform_tgt, projection=projection_tgt)

    # Pass cloud mask
    scl = ds_tgt.isel(time=i).s2_SCL
    scl_mask = xr.where(scl.isin([0,1,2,3,7,8,9,10]), True, False) 
    cloud = ds_tgt.isel(time=i).s2_mask
    cloud_mask = xr.where(cloud != 0, True, False)
    data_mask = scl_mask & cloud_mask

    #print(f'Clouds masked {data_mask.values.sum()}/{data_mask.shape[0]*data_mask.shape[1]}. Missing data: {np.sum(target_image.values == 65535)}')
    if not data_mask.values.sum() > 0.8*(data_mask.shape[0]*data_mask.shape[1]):  # max 80% clouds
      try:
        # Initialize the COREG object
        CR = COREG(
            im_ref=geo_ref_image,  # Reference image array
            im_tgt=geo_tgt_image,  # Target image array
            ws=(128,128),          # Size of the matching window in pixels
            max_iter=10,            # Maximum number of iterations
            path_out=None,         # Path to save the coregistered image (None if not saving)
            fmt_out='Zarr',        # Output format (None if not saving)
            nodata =(255, 65535),
            mask_baddata_tgt=data_mask.values,
            footprint_poly_ref=footprint_ref,
            footprint_poly_tgt=footprint_tgt,
            align_grids=True,
            q=True,
            max_shift=3
        )

        # Compute shifts
        CR.calculate_spatial_shifts()
        corrected_dict = CR.correct_shifts() # returns an OrderedDict containing the co-registered numpy array and its corresponding geoinformation.
        shift_x, shift_y = CR.coreg_info['corrected_shifts_px']['x'],  CR.coreg_info['corrected_shifts_px']['y']
        #geo_tgt_image = GeoArray(ds_tgt.isel(time=i).to_array().values.transpose(1, 2, 0), geotransform=geotransform_tgt, projection=projection_tgt) #lat, lon, band
        #corrected_image = apply_shifts(geo_tgt_image.arr, [shift_x, shift_y], geo_tgt_image.arr.shape)
      
        return (shift_x, shift_y), True  # Return coreg shifts and a flag indicating success

      except Exception as e:
        #print(f'Error in coreg step {i}: {e}')
        return None, False  # Return the original image and failure flag

    else:
      return None, False


def coreg_dask(ds, ref):
  
  # Remove variables that shouldn't be coregistered
  to_drop = ['mean_sensor_azimuth', 'mean_sensor_zenith', 'mean_solar_azimuth', 'mean_solar_zenith', 'product_uri']
  ds_tgt = ds.drop_vars(to_drop)
  bands = list(ds_tgt.data_vars)

  # Select band for coreg
  ref = ref['R']

  # Convert ref and target to GeoArray with some geo information
  pixel_width_ref = 0.1 #abs(ref.x[1] - ref.x[0]).item()
  pixel_height_ref = 0.1 #abs(ref.y[1] - ref.y[0]).item()
  geotransform_ref = (ref.x.min().item(), pixel_width_ref, 0, ref.y.max().item(), 0, -pixel_height_ref)
  projection_ref = CRS.from_epsg(32632).to_wkt()
  geo_ref_image = GeoArray(ref.values.astype(np.float32), geotransform=geotransform_ref, projection=projection_ref)
  minx, miny, maxx, maxy = [ref.x.min(), ref.y.min()-pixel_height_ref, ref.x.max()+pixel_width_ref, ref.y.max()]
  footprint_ref = Polygon([(maxx, maxy), (maxx, miny), (minx, miny), (minx, maxy), (maxx, maxy)])

  pixel_width_tgt = abs(ds_tgt.lon[1] - ds_tgt.lon[0]).item()
  pixel_height_tgt = abs(ds_tgt.lat[1] - ds_tgt.lat[0]).item()
  geotransform_tgt = (ds_tgt.lon.min().item(), pixel_width_tgt, 0, ds_tgt.lat.max().item(), 0, -pixel_height_tgt)
  projection_tgt = CRS.from_epsg(32632).to_wkt()
  minx, miny, maxx, maxy = [ds_tgt.lon.min(), ds_tgt.lat.min()-pixel_height_tgt, ds_tgt.lon.max()+pixel_width_tgt, ds_tgt.lat.max()]
  footprint_tgt = Polygon([(maxx, maxy), (maxx, miny), (minx, miny), (minx, maxy), (maxx, maxy)])

  # Global coregistration
  corrected_images_stack = []
  coreg_mask = np.zeros(len(ds_tgt.time), dtype=bool)

  tasks = [coreg_single_step(i, ds_tgt, geo_ref_image, geotransform_tgt, projection_tgt, footprint_tgt, geotransform_ref, projection_ref, footprint_ref, coreg_mask)
         for i in range(ds_tgt.sizes['time'])]

  # Trigger computation in parallel
  results = dask.compute(*tasks)

  # Extract results from Dask output
  shifts, coreg_mask = zip(*results)

  """
  corrected_images_stack = np.stack(corrected_images_stack, axis=0).transpose((3,0,1,2))  # Final shape is (bands, time, lat, lon)

  # Create a new xarray Dataset
  time_dim = ds_tgt.sizes['time']
  lat_dim = ds_tgt.sizes['lat']
  lon_dim = ds_tgt.sizes['lon']
  bands_dim = len(bands)

  # Create DataArray for each band
  data_vars = {band: xr.DataArray(
      data=corrected_images_stack[bands.index(band),:, :, :],
      dims=['time','lat','lon'],
      coords={'lon': ds_tgt['lon'].values, 'lat': ds_tgt['lat'].values, 'time': ds_tgt['time'].values},
      name=band
  ) for band in bands}

  # Create Dataset
  ds_coreg = xr.Dataset(data_vars=data_vars)

  # Convert back to uint16
  ds_coreg = ds_coreg.fillna(65535).clip(0, 65535).round()
  ds_coreg = ds_coreg.astype(np.uint16)

  # Add back variables
  ds_coreg[to_drop] = ds[to_drop]
 
  return ds_coreg, coreg_mask
  """
  return shifts, coreg_mask


def extract_date(time):
    """ 
    Extract year, month and day as int from numpy datetime64[ns] object

    :param time: numpy datetime64[ns] object
    :return: year, month, day
    """
    year = time.astype('datetime64[Y]').astype(int) + 1970
    month = (time.astype('datetime64[M]').astype(int) % 12) + 1
    day = (time.astype('datetime64[D]').astype(int) -
        time.astype('datetime64[M]').astype('datetime64[D]').astype(int)) + 1

    # Format month and day with leading zeros
    month_str = f"{month:02d}"
    day_str = f"{day:02d}"

    return year, month_str, day_str


def split_and_save(ds, minx, maxy, output_folder, attrs):
    """
    Extract cube of interest, update attributes and save data year by year

    :param ds: xr Dataset with coregistered data
    :param minx: min lon of cube
    :param maxy: max lat of cube
    :param output_folder: folder where to write new data
    :param attrs: original metadata of cube
    """
    if ds.lat.values[0] > ds.lat.values[-1]:
      ds = ds.sel(lat=slice(maxy,maxy-1270), lon=slice(minx, minx+1270))
    else:
      ds = ds.sel(lat=slice(maxy-1270, maxy), lon=slice(minx, minx+1270))
      ds = ds.sel(lat=slice(None, None, -1))
    
    
    for yr in np.arange(2017, 2024, 1):

        mask_dates = np.ones(len(ds.time), dtype=bool)
        dates_to_drop = [i for i, date in enumerate(ds.time.values) if date.astype('datetime64[Y]').astype(int) + 1970 != yr] 
        mask_dates[dates_to_drop] = False
        ds_yr = ds.isel(time=mask_dates)

        if len(ds_yr.time):
          time_min, time_max = ds_yr.time.values[0], ds_yr.time.values[-1]
          year_start, month_start, day_start = extract_date(time_min)
          year_end, month_end, day_end = extract_date(time_max)

          attrs['history'] += f". Coregistered with SwissImage on {datetime.date.today()}."
          ds_yr.attrs = attrs
          
          # Save to zarr store
          output_path = os.path.join(output_folder, f'S2_{minx}_{maxy}_{year_start}{month_start}{day_start}_{year_end}{month_end}{day_end}.zarr')
          compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
          if not os.path.exists(output_path):
            print('Saving', output_path)
            ds_yr.to_zarr(output_path, consolidated=True, mode='w', encoding={var: {'compressor': compressor} for var in ds_yr.data_vars})
        
          
    return


def run_coregistration_file(f, target_folder, reference_folder, output_folder):
  """
  Run coregistration pipeline for all files in target folder. Files in target and reference must have same name system and contain zarr stores

  :param f: file to coregister
  :param reference_folder: path to reference files
  :param output_folder: folder where to write new files
  """

  # Load file and all possible contigous files (up to 8 other cubes)
  ds, minx, maxy, attrs = load_cubes(f, target_folder)
  ds = ds.isel(lat=slice(None, None, -1)) # make sure lat is decreasing
  # Load SwissImage of correspoding central cube
  ref = xr.open_zarr(os.path.join(reference_folder, f'SwissImage0.1_{int(minx)}_{int(maxy)}.zarr')).compute() # make sure y is decreasing
  # Coreg (if too big, can do year by year)
  shifts, coreg_mask = coreg_dask(ds, ref)

  return shifts, coreg_mask, ds


def parallel_process(target_folder, reference_folder, output_folder, num_workers=1):

    if not os.path.exists(output_folder):
      os.makedirs(output_folder)

    # Filter files
     
    target_files = [os.path.join(target_folder, f) for f in os.listdir(target_folder) if f.endswith('.zarr')]
    processed_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.zarr')]
    """
    reckenholz_files = ['S2_462300_5254060_20170107_20171231.zarr', 'S2_462300_5252780_20170107_20171231.zarr', 'S2_463580_5254060_20170107_20171231.zarr', 'S2_463580_5252780_20170107_20171231.zarr']
    target_files = [os.path.join(target_folder, f) for f in os.listdir(target_folder) if f in reckenholz_files]
    processed_files = []
    """
    
    tasks = [(f, processed_files, reference_folder, output_folder) for f in target_files if f not in processed_files]

    # Using ProcessPoolExecutor to process files in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks to the executor
        future_to_task = {executor.submit(run_coregistration_file, *task): task for task in tasks}
        
        # Iterate through completed futures
        for future in as_completed(future_to_task):
            task = future_to_task[future]  # Get the associated task
            try:
                result = future.result()  # Get the result of the future (raises exceptions if any)
                f = task[0]
                print(f"Successfully processed {f}")
            except Exception as e:
                f = task[0]  # Get the file being processed in the task
                print(f"Error occurred with file {f}: {e}")
                # Optionally, log or take some other action here
                break  # Exit the loop on error, or you can continue if desired

              
def update(frame):
    # Update images for the current timestamp
    im_rgb.set_array(rgb[frame].values*3)
    im_rgbcoreg.set_array(rgb_coreg[frame].values*3)
    
    return im_rgb, im_rgbcoreg


def plot_mp4(ref, ds, ds_coreg, outpath):
  global rgb, rgb_coreg, im_rgb, im_rgbcoreg
  scale_factor = 1.0 / 10000.0  # Scale factor for DN to [0, 1]
  r = ds_coreg['s2_B04'] * scale_factor
  g = ds_coreg['s2_B03'] * scale_factor
  b = ds_coreg['s2_B02'] * scale_factor

  # Stack bands into an RGB array
  rgb_coreg = xr.concat([r, g, b], dim='band').transpose('time', 'lat', 'lon', 'band')
  rgb_coreg = rgb_coreg.where(~np.isnan(rgb_coreg), other=1.0)

  r = ds['s2_B04'] * scale_factor
  g = ds['s2_B03'] * scale_factor
  b = ds['s2_B02'] * scale_factor

  # Stack bands into an RGB array
  rgb = xr.concat([r, g, b], dim='band').transpose('time', 'lat', 'lon', 'band')
  rgb = rgb.where(~np.isnan(rgb), other=1.0)

  ref_rgb = ref[['R','G','B']].to_array().values.transpose(1,2,0)

  # Set up the figure and axis
  fig, axs = plt.subplots(1, 3, figsize=(15, 5))

  # Initialize images
  img_ref = axs[0].imshow(ref_rgb)
  im_rgb = axs[1].imshow(rgb[0].values*3)
  im_rgbcoreg = axs[2].imshow(rgb_coreg[0].values*3)

  # Set titles and axis labels
  axs[0].set_title('Ref')
  axs[1].set_title('Orig')
  axs[2].set_title('Coreg')

  # Create the animation
  ani = FuncAnimation(fig, update, frames=len(rgb.time), blit=True)

  # Save the animation as an MP4 video using ffmpeg
  ani.save(outpath, writer='ffmpeg', fps=2)  # Adjust fps (frames per second) as needed

  return



if __name__ == "__main__":

  target_folder = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH')
  reference_folder = os.path.expanduser('~/mnt/eo-nas1/data/swisstopo/SwissImage/cubes/10cm')
  output_folder = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/coreg/CH')

  
  # Step 1: compute shifts at one location

  shift_dict = {}

  tiles = {'32TMT': [(7.66286529045287, 47.845914826827),(7.68758850048273, 46.8581759574933),(9.12805716115049, 46.8656282472998),(9.1304703619161, 47.8536277114283),(7.66286529045287, 47.845914826827)],
            #'32TLT': [(6.32792675947954,47.8225951267953),(6.37728036218156, 46.8356437136561),(7.81664846692137, 46.8595830833696),(7.79435498889756, 47.8473711518664),(6.32792675947954, 47.8225951267953)],
            #'32TNT': [(8.9997326423426, 47.8537018420053),(8.99973758744939,46.8656998728757),(10.4401498244026, 46.8566398814282),(10.4672773767603, 47.8443250460311),(8.9997326423426, 47.8537018420053)],
            #'32TNS': [(8.99973715756222,46.9537091787344),(8.9997418700468, 45.9655511480415),(10.4166561015996, 45.9567698586004),(10.442508097938, 46.9446214409232),(8.99973715756222, 46.9537091787344)],
            #'32TMS': [(7.68543924709582, 46.9461622205863),(7.7089998702566, 45.9582586884214),(9.12596726307512, 45.9654817261501),(9.12826694512377, 46.9536373337673),(7.68543924709582, 46.9461622205863)],
            #'32TLS': [(6.37298980762373, 46.9235610080068),(6.42002507864337, 45.9364192287437),(7.83595551698163, 45.9596225320207),(7.81471043979331, 46.94757365544),(6.37298980762373, 46.9235610080068)],
            #'31TGM': [(5.62648535526562, 46.9235730577955),(5.57945945554239, 45.9364308725672),(6.99300216652316, 45.8957352511058),(7.06565713267192, 46.8814596607493),(5.62648535526562, 46.9235730577955)],
            #'31TGN': [(5.67153942638492, 47.8226075594756),(5.62219565549863, 46.8356557266896),(7.05902986502132, 46.793670687313),(7.13525847525188, 47.7791570713891),(5.67153942638492, 47.8226075594756)]
          }
  
  for tile, tile_geom in tiles.items():
    polygon = Polygon(tile_geom)
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs='EPSG:4326').to_crs('EPSG:32632')
    
    # Find grid cubes in tile 
    yrs = [2017]
    tile_cubes = find_cubes_aoi_yrs(data_folder=target_folder, geom=gdf, yrs=yrs)
    tile_cubes = tile_cubes.set_crs(32632)
    

    # Sample grid cubes
    sample_cubes = tile_cubes.iloc[[100,400,2000]]#.sample(3)
    """ 
    f, ax = plt.subplots()
    tile_cubes.plot(ax=ax, alpha=0.5)
    sample_cubes.iloc[0:1].plot(ax=ax, color='r')
    sample_cubes.iloc[1:].plot(ax=ax, color='g')
    cx.add_basemap(ax=ax, crs=tile_cubes.crs)
    plt.savefig('tile_cube.png')
    
   
   
    # Launch coreg without applying shift, just saving shifts
    print(sample_cubes.file.values[0])
    shifts, coreg_mask, ds = run_coregistration_file(sample_cubes.file.values[0], target_folder, reference_folder, output_folder)
    lon_size, lat_size, time_size = ds.sizes['lon'], ds.sizes['lat'], ds.sizes['time']
    
    shift_dict[f'{sample_cubes.file.values[0]}'] = {
      'timestamps' : ds.time.values,
      'product_uri' : ds.product_uri.values[lon_size//2][lat_size//2],
      'shifts' : shifts
    }
    
    # Save shifts: file, timestamp, product_uri, shifts
    with open('shift_dict.pkl', 'wb') as f:
      pickle.dump(shift_dict, f)
    
    """ 
    
    
    break
  

  # Step 2: apply saved shifts
  with open('shift_dict.pkl', 'rb') as f:
    shift_dict = pickle.load(f)

  file_key = f'{sample_cubes.file.values[0]}'
  timestamps = shift_dict[file_key]['timestamps']
  product_uris = shift_dict[file_key]['product_uri']
  shifts = shift_dict[file_key]['shifts']

   
  for i, cube in sample_cubes.iterrows():
    ds, minx, maxy, attrs = load_cubes(cube.file, target_folder)
    ds = ds.isel(lat=slice(None, None, -1)) 

    to_drop = ['mean_sensor_azimuth', 'mean_sensor_zenith', 'mean_solar_azimuth', 'mean_solar_zenith', 'product_uri']
    ds_tgt = ds.drop_vars(to_drop)
    bands = list(ds_tgt.data_vars)

    coreg_stack = []
    # loop over time and apply shift where needed
    print(ds.sizes)
    for i in range(ds.sizes['time']):
      t = ds.time.values[i]
      if t in timestamps:
        idx = list(timestamps).index(t) # to do: what if there are duplicate timestamps
        corresponding_product_uri = product_uris[idx]
        corresponding_shifts = shifts[idx]
        
        # Print or process the corresponding product_uri and shifts
        print(f'Found match for timestamp {t}:')
        print(f'  Product URI: {corresponding_product_uri}')
        print(f'  Shifts: {corresponding_shifts}')

        # Apply shifts
        tgt_image = ds_tgt.isel(time=i).to_array().values.transpose(1, 2, 0) #lat, lon, band
        corrected_image = apply_shifts(tgt_image, [corresponding_shifts[0], corresponding_shifts[1]], tgt_image.shape)
        coreg_stack.append(corrected_image)

        break

    break 


  
  # Step 3: Compare new coregistration



