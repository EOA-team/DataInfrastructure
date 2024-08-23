""" 
Coregistration pipeline using SwissImage to correct S2 data
"""

import os
import xarray as xr
import numpy as np
import glob
import warnings
warnings.filterwarnings('ignore')
from geoarray import GeoArray
from arosics import COREG
from arosics import DESHIFTER
from pyproj import CRS
from shapely import Polygon
import zarr
import datetime
import pandas as pd
from collections import defaultdict
import rioxarray


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
  for (timestamp, tile_id, granule_id), datasets in grouped_datasets.items():
      if len(datasets) > 1:
          combined_ds = xr.combine_by_coords(datasets, combine_attrs='override')
          combined_ds[['mean_sensor_azimuth', 'mean_sensor_zenith','mean_solar_azimuth', 'mean_solar_zenith', 'product_uri']] = \
            datasets[0][['mean_sensor_azimuth', 'mean_sensor_zenith','mean_solar_azimuth', 'mean_solar_zenith', 'product_uri']]
      else:
          combined_ds = datasets[0]
      
      combined_datasets.append(combined_ds)
  
  ds = xr.concat(combined_datasets, dim="time")

  return ds


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
  """
  # Calculate contiguous cubes
  xs = np.arange(minx-1280, minx+1280*2, 1280)
  ys = np.arange(maxy-1280, maxy+1280*2, 1280)

  # Find files that have these coords (all years)
  file_patterns = [os.path.join(target_folder, f'S2_{x}_{y}_*.zarr') for x in xs for y in ys]
  cubes = [file for pattern in file_patterns for file in glob.glob(pattern)]
  print(cubes)
  """
  cubes = ['/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5113260_20170103_20171231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5113260_20210102_20211230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5113260_20230102_20231230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5113260_20180103_20181231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5113260_20220102_20221230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5113260_20190103_20191231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5113260_20200103_20201230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5114540_20170103_20171231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5114540_20210102_20211230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5114540_20230102_20231230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5114540_20180103_20181231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5114540_20220102_20221230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5114540_20190103_20191231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5114540_20200103_20201230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5115820_20170103_20171231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5115820_20210102_20211230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5115820_20230102_20231230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5115820_20190103_20191231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5115820_20180103_20181231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5115820_20220102_20221230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5115820_20200103_20201230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5113260_20170103_20171231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5113260_20210102_20211230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5113260_20230102_20231230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5113260_20180103_20181231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5113260_20220102_20221230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5113260_20190103_20191231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5113260_20200103_20201230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5114540_20170103_20171231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5114540_20210102_20211230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5114540_20230102_20231230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5114540_20180103_20181231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5114540_20220102_20221230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5114540_20190103_20191231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5114540_20200103_20201230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5115820_20170103_20171231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5115820_20210102_20211230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5115820_20220102_20221230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5115820_20230102_20231230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5115820_20180103_20181231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5115820_20190103_20191231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5115820_20200103_20201230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5113260_20170103_20171231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5113260_20210102_20211230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5113260_20230102_20231230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5113260_20180103_20181231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5113260_20220102_20221230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5113260_20190103_20191231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5113260_20200103_20201230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5114540_20170103_20171231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5114540_20210102_20211230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5114540_20230102_20231230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5114540_20180103_20181231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5114540_20220102_20221230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5114540_20190103_20191231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5114540_20200103_20201230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5115820_20170103_20171231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5115820_20210102_20211230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5115820_20220102_20221230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5115820_20230102_20231230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5115820_20190103_20191231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5115820_20180103_20181231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5115820_20200103_20201230.zarr']
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


def coreg(ds, ref):
  """
  Apply coregistration usign AROSICs package

  :param ds: xr.Dataset with data to coregister
  :param ref: reference image
  :return ds_coreg: coregistered dataset
  """
  # Remove variables that shouldn't be coregistered
  to_drop = ['mean_sensor_azimuth', 'mean_sensor_zenith', 'mean_solar_azimuth', 'mean_solar_zenith', 'product_uri']
  ds_tgt = ds.drop_vars(to_drop)
  bands = list(ds_tgt.data_vars)

  # Select band for coreg
  ref = ref['R']
  # Reorder the variables so that s2_B04 is the first variable
  variables = ['s2_B04'] + [var for var in ds_tgt.data_vars if var != 's2_B04']
  ds_tgt = ds_tgt[variables]

  # Convert ref and target to GeoArray with some geo information
  pixel_width_ref = abs(ref.x[1] - ref.x[0]).item()
  pixel_height_ref = abs(ref.y[1] - ref.y[0]).item()
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

  for i in range(ds.sizes['time']):
      
      target_image = ds.isel(time=i)
      geo_tgt_image = GeoArray(target_image.to_array().values.transpose(1, 2, 0), geotransform=geotransform_tgt, projection=projection_tgt)
      # Pass cloud mask
      scl = ds.isel(time=i).s2_SCL
      scl_mask = xr.where(scl.isin([0,1,2,3,7,8,9,10]), True, False) 
      cloud = ds.isel(time=i).s2_mask
      cloud_mask = xr.where(cloud != 0, True, False)
      data_mask = scl_mask & cloud_mask
      
      # Check if there is any data
      #print(f'Clouds masked {data_mask.values.sum()}/{data_mask.shape[0]*data_mask.shape[1]}')
      #print(f'Missing data: {np.sum(target_image.values == 65535)}')
      if not data_mask.values.sum() == data_mask.shape[0]*data_mask.shape[1]: 
        try:

            # Initialize the COREG object
            CR = COREG(
                im_ref=geo_ref_image,  # Reference image array
                im_tgt=geo_tgt_image, #target_image.to_array().values.transpose(1, 2, 0),     # Target image array
                ws=(80,80),           # Size of the matching window
                max_iter=10,             # Maximum number of iterations
                path_out=None,           # Path to save the coregistered image (None if not saving)
                fmt_out='Zarr',             # Output format (None if not saving)
                nodata = (255, 65535),
                mask_baddata_tgt= data_mask.values,
                footprint_poly_ref=footprint_ref,
                footprint_poly_tgt=footprint_tgt,
                align_grids=True,
                #s_b4match = 1, # target image band to use (starts with 1)
                #r_b4match = 3, # ref image band to use (starts with 1)
            )

            # Compute shifts
            CR.calculate_spatial_shifts()
            corrected_dict = CR.correct_shifts() # returns an OrderedDict containing the co-registered numpy array and its corresponding geoinformation.
            corrected_images_stack.append(corrected_dict['arr_shifted'])
            print(corrected_dict['arr_shifted'])
        
        except Exception as e:
            print(f'Error: {e}')
            corrected_images_stack.append(geo_tgt_image.arr)
            pass
  
  corrected_images_stack = np.stack(corrected_images_stack, axis=0).transpose((3,0,1,2))  # Final shape is (bands, time, lat, lon)

  # Create a new xarray Dataset
  time_dim = ds.sizes['time']
  lat_dim = ds.sizes['lat']
  lon_dim = ds.sizes['lon']
  bands_dim = len(bands)

  # Create DataArray for each band
  data_vars = {band: xr.DataArray(
      data=corrected_images_stack[bands.index(band),:, :, :],
      dims=['time','lat','lon',],
      coords={'lon': ds['lon'].values, 'lat': ds['lat'].values, 'time': ds['time'].values},
      name=band
  ) for band in bands}

  # Create Dataset
  ds_coreg = xr.Dataset(data_vars=data_vars)
  ds_coreg[to_drop] = ds[to_drop]


  return ds_coreg


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
    ds = ds.sel(lat=slice(maxy-1270,maxy), lon=slice(minx, minx+1270))
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
          output_path = output_folder + f'S2_{minx}_{maxy}_{year_start}{month_start}{day_start}_{year_end}{month_end}{day_end}.zarr'
          compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
          if not os.path.exists(output_path):
              ds_yr.to_zarr(output_path, consolidated=True, mode='w', encoding={var: {'compressor': compressor} for var in ds_yr.data_vars})
          
    return


def run_coregistration(target_folder, reference_folder, output_folder):
  """
  Run coregistration pipeline for all files in target folder. Files in target and reference must have same name system and contain zarr stores

  :param target folder: path to target files
  :param reference_folder: path to reference files
  :param output_folder: folder where to write new files
  """
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  target_files = [os.path.join(target_folder, f) for f in os.listdir(target_folder) if f.endswith('.zarr') and '263900_5114540' in f] #

  for i, f in enumerate(target_files):
    print(f)
    # Load file and all possible contigous files (up to 8 other cubes)
    ds, minx, maxy, attrs = load_cubes(f, target_folder)
    # Load SwissImage of correspoding central cube
    ref = xr.open_zarr(os.path.join(reference_folder, f'SwissImage0.1_{int(minx)}_{int(maxy)}.zarr')).compute()
    # Clean ds (clouds, missing dates...)
    #ds = clean_dataset(ds, cloud_thresh=0.1, snow_thresh=0.1, shadow_thresh=0.1, cirrus_thresh=1000)  # could pass thresholds for cleaning
    # Coreg (if too big, can do year by year)
    ds_coreg = coreg(ds.isel(time=slice(0,2)), ref)
    # Save year by year
    split_and_save(ds_coreg, minx, maxy, output_folder, attrs)
    break



if __name__ == "__main__":

  target_folder = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH')
  reference_folder = os.path.expanduser('~/mnt/eo-nas1/data/swisstopo/SwissImage/cubes/10cm')
  output_folder = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/coreg/CH')

  run_coregistration(target_folder, reference_folder, output_folder)