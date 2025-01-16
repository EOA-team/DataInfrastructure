import dask
import os
import xarray as xr
import numpy as np
import geopandas as gpd
import glob
import warnings
warnings.filterwarnings('ignore')
from geoarray import GeoArray
from arosics import COREG
from pyproj import CRS
from shapely import Polygon, box
import zarr
import pandas as pd
from collections import defaultdict
from scipy.ndimage import affine_transform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import time
import rioxarray
import matplotlib.pyplot as plt


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


def find_cubes_aoi(data_folder, geom):
  """
  Find S2 cubes that fall in perimeter given by geometry

  :param data_folder: path to where S2 data is stored
  :param geom: gpd.GeoDataFrame containing geometry to use to filter files
  :return: gdf with filtered files as rows
  """
  cubes = [f for f in os.listdir(data_folder) if f.endswith('zarr')]
  df_cubes = pd.DataFrame(cubes, columns=['file'])
  df_cubes[['minx', 'miny', 'maxx', 'maxy', 'yr_start', 'yr_end']] = df_cubes['file'].apply(lambda x: pd.Series(extract_bounds_multiyear(x)))
  df_cubes['geometry'] = df_cubes.apply(lambda row: box(row['minx'], row['miny'], row['maxx'], row['maxy']), axis=1)
  gdf_cubes = gpd.GeoDataFrame(df_cubes, geometry='geometry')
  filtered_files = gdf_cubes[gdf_cubes.intersects(geom.unary_union)]
  filtered_files = filtered_files.drop_duplicates(subset=['minx', 'miny', 'maxx', 'maxy'], keep='first')
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
  file_pattern = os.path.join(target_folder, f'S2_{minx}_{maxy}_*.zarr')
  cubes = [file for file in glob.glob(file_pattern)]
  #cubes = ['/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5113260_20170103_20171231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5113260_20210102_20211230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5113260_20230102_20231230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5113260_20180103_20181231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5113260_20220102_20221230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5113260_20190103_20191231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5113260_20200103_20201230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5114540_20170103_20171231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5114540_20210102_20211230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5114540_20230102_20231230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5114540_20180103_20181231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5114540_20220102_20221230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5114540_20190103_20191231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5114540_20200103_20201230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5115820_20170103_20171231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5115820_20210102_20211230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5115820_20230102_20231230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5115820_20190103_20191231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5115820_20180103_20181231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5115820_20220102_20221230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_262620_5115820_20200103_20201230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5113260_20170103_20171231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5113260_20210102_20211230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5113260_20230102_20231230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5113260_20180103_20181231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5113260_20220102_20221230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5113260_20190103_20191231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5113260_20200103_20201230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5114540_20170103_20171231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5114540_20210102_20211230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5114540_20230102_20231230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5114540_20180103_20181231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5114540_20220102_20221230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5114540_20190103_20191231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5114540_20200103_20201230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5115820_20170103_20171231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5115820_20210102_20211230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5115820_20220102_20221230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5115820_20230102_20231230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5115820_20180103_20181231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5115820_20190103_20191231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_263900_5115820_20200103_20201230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5113260_20170103_20171231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5113260_20210102_20211230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5113260_20230102_20231230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5113260_20180103_20181231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5113260_20220102_20221230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5113260_20190103_20191231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5113260_20200103_20201230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5114540_20170103_20171231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5114540_20210102_20211230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5114540_20230102_20231230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5114540_20180103_20181231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5114540_20220102_20221230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5114540_20190103_20191231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5114540_20200103_20201230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5115820_20170103_20171231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5115820_20210102_20211230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5115820_20220102_20221230.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5115820_20230102_20231230.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5115820_20190103_20191231.zarr', '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5115820_20180103_20181231.zarr','/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_265180_5115820_20200103_20201230.zarr']
  
  ds = xr.open_mfdataset(cubes, combine='by_coords').compute()
  
  cube_attrs = ds.attrs 
 
  return ds, minx, maxy, cube_attrs


# Parallelize each coregistration step with Dask
@dask.delayed
def coreg_single_step(i, ds_tgt, geo_ref_image, geotransform_tgt, projection_tgt, footprint_tgt, geotransform_ref, projection_ref, footprint_ref):
    
    # Same coreg logic for one timestep
    target_image = ds_tgt.isel(time=i).s2_B04
    geo_tgt_image = GeoArray(target_image.values, geotransform=geotransform_tgt, projection=projection_tgt)

    # Pass cloud mask as nodata mask
    scl = ds_tgt.isel(time=i).s2_SCL
    scl_mask = xr.where(scl.isin([0,1,7,8,9,10]), True, False) 
    cloud = ds_tgt.isel(time=i).s2_mask
    cloud_mask = xr.where(cloud.isin([1,2,4]), True, False)  #cloud != 0, True, False)
    data_mask = scl_mask & cloud_mask
    geo_data_mask = GeoArray(data_mask.values, geotransform=geotransform_tgt, projection=projection_tgt)

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
          mask_baddata_tgt=geo_data_mask, #data_mask.values,
          footprint_poly_ref=footprint_ref,
          footprint_poly_tgt=footprint_tgt,
          align_grids=True,
          q=True,
          max_shift=3
      )

      # Compute shifts
      CR.calculate_spatial_shifts()
      """
      from py_tools_ds.geo.coord_trafo import mapXY2imXY
      wp = tuple(CR.win_pos_XY)
      imX, imY = mapXY2imXY(wp, CR.shift.mask_baddata.gt)
      print(CR.shift.mask_baddata[int(imY), int(imX)])
      """
      corrected_dict = CR.correct_shifts() # returns an OrderedDict containing the co-registered numpy array and its corresponding geoinformation.
      shift_x, shift_y = CR.coreg_info['corrected_shifts_px']['x'],  CR.coreg_info['corrected_shifts_px']['y']

      return (shift_x, shift_y), True  # Return coreg shifts and a flag indicating success

    except Exception as e:
      #print(f'Error in coreg step {i}: {e}')
      #print('Skipped', data_mask.sum().item())
      return (np.nan, np.nan), False  # Return the original image and failure flag


def coreg(ds, ref, batch_size=20):
  
  start = time.time()
  # Remove variables that shouldn't be coregistered
  to_drop = ['mean_sensor_azimuth', 'mean_sensor_zenith', 'mean_solar_azimuth', 'mean_solar_zenith', 'product_uri']
  ds_tgt = ds.drop_vars(to_drop)
  bands = list(ds_tgt.data_vars)

  # Select band for coreg
  ref = ref['R']

  # Convert ref and target to GeoArray with some geo information
  pixel_width_ref = 0.1 
  pixel_height_ref = 0.1 
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

  # Check which timestamps are cloud free (<60% clouds)
  scl = ds_tgt.s2_SCL
  scl_mask = xr.where(scl.isin([0,1,7,8,9,10]), True, False) 
  cloud = ds_tgt.s2_mask
  cloud_mask = xr.where(cloud.isin([1,2,4]), True, False)  #cloud != 0, True, False)
  data_mask = scl_mask & cloud_mask
  cloud_cover = data_mask.sum(dim=["lon", "lat"])
  total_cells = ds_tgt.sizes['lat'] * ds_tgt.sizes['lon']
  cloud_cover = cloud_cover/total_cells
  cloud_mask = cloud_cover < 0.6

  # Parallel processing  
  tasks = [coreg_single_step(i, ds_tgt, geo_ref_image, geotransform_tgt, projection_tgt, footprint_tgt, geotransform_ref, projection_ref, footprint_ref)
         for i in range(ds_tgt.sizes['time']) if cloud_mask.values[i]]
  end = time.time()
  print('Preparing coreg', end-start)

  final_results = []
  for i in range(0, len(tasks), batch_size):
    start = time.time()
    batch = tasks[i:i + batch_size]
    results = dask.compute(*batch)
    final_results.extend(results)
    end = time.time()
    #print(f'Batch of {batch_size} took', end-start)

  # Extract results from Dask output
  shifts_done, coreg_mask_done = zip(*final_results)

  # Return ds_tgt where cloud_mask and coreg_mask done
  ds = ds.where(cloud_mask, drop=True)
  #coreg_mask_done = list(coreg_mask_done) + [False]*(len(ds.time)-len(coreg_mask_done))
  ds = ds.isel(time=list(coreg_mask_done))

  return shifts_done, coreg_mask_done, ds


def run_coregistration_file(f, target_folder, reference_folder, batch_size):
  """
  Run coregistration pipeline for all files in target folder. Files in target and reference must have same name system and contain zarr stores

  :param f: file to coregister
  :param target_folder: path to target files
  :param reference_folder: path to reference files
  :param batch_size: number of timestamps to process in parallel (depends on your compute power)
  """

  # Load file, inlcuding other years at same location
  start = time.time()
  ds, minx, maxy, attrs = load_cubes(f, target_folder)
  if ds.lat.values[0] < ds.lat.values[-1]:
    ds = ds.isel(lat=slice(None, None, -1)) 
  end = time.time()
  print('Loading cubes', end-start)

  start = time.time()
  # Load SwissImage of correspoding central cube
  ref = xr.open_zarr(os.path.join(reference_folder, f'SwissImage0.1_{int(minx)}_{int(maxy)}.zarr')).compute() 
  if ref.y.values[0] < ref.y.values[-1]:
    ref = ref.isel(y=slice(None, None, -1)) 
  end = time.time()
  print('Loading SwissImage', end-start)

  # Coreg
  start = time.time()
  shifts, coreg_mask, ds = coreg(ds, ref, batch_size)
  end = time.time()
  print('Coreg took', end-start)

  return shifts, coreg_mask, ds


def process_single_file(f, target_folder, reference_folder, batch_size):
    """
    Process a single file for coregistration.

    :param task: Tuple containing (file_path, target_folder, reference_folder)
    :return: Processed DataFrame for the file
    """
    try:
        shifts, coreg_mask, ds = run_coregistration_file(f, target_folder, reference_folder, batch_size)
        
        # Extract information
        lon_size, lat_size, time_size = ds.sizes['lon'], ds.sizes['lat'], ds.sizes['time']
        cube_key = f"{os.path.basename(f).split('_')[1]}_{os.path.basename(f).split('_')[2]}"
        uris = ds.isel(lon=lon_size // 2, lat=lat_size // 2).product_uri.compute().values
        
        rows = []
        for i, uri in enumerate(uris):
            if not coreg_mask[i]:
                rows.append([cube_key, uri, np.nan, np.nan])
            else:
                rows.append([cube_key, uri, shifts[i][0], shifts[i][1]])

        rows_array = np.array(rows, dtype=object)
        df = pd.DataFrame(rows_array, columns=["name", "uri", "shift_x", "shift_y"])
        return df
    except Exception as e:
        print(f"Error processing file {f}: {e}")
        return None
        

def save_to_pickle(dataframe, tile_name): #, lock):
    """
    Save DataFrame to a pickle file or add to existing file.
    """
    dataframe = dataframe.dropna(subset=['shift_x', 'shift_y'], how='all')
    pickle_filename = f"shift_results_{tile_name}.pkl"
    #with lock:  # Ensure thread-safe writes
    if not os.path.exists(pickle_filename): 
        dataframe.to_pickle(pickle_filename)
        print(f"Saved DataFrame to {pickle_filename}, {len(dataframe)}")
    else:
        # Load existing file, append, and save
        existing_df = pd.read_pickle(pickle_filename)
        updated_df = pd.concat([existing_df, dataframe], ignore_index=True)
        updated_df.to_pickle(pickle_filename)
        print(f"Saved DataFrame to {pickle_filename}, {len(updated_df)}")
    return


def process_tile(file_list, processed_names, target_folder, reference_folder, tile_name, batch_size):
    """
    Launch coregistration of files in parallel with batching.

    :param file_list: List of files to coregister.
    :param processed_names: List of processed file names.
    :param target_folder: Directory where file_list is stored (files to process).
    :param reference_folder: Directory where reference images are stored.
    :param tile_name: Name of the S2 tile in which the files are located.
    """
    # Filter files
    target_files = [os.path.join(target_folder, f) for f in os.listdir(target_folder) if f in file_list]
    target_files = [f for f in target_files if not any(name in f for name in processed_names)]

    for i, f in enumerate(target_files):
      print(f"----Processing file {i+1}/{len(target_files)}")
      start = time.time()
      result_df = process_single_file(f, target_folder, reference_folder, batch_size)
      end = time.time()
      print('Whole process for file took', end-start)
      if result_df is not None:
        save_to_pickle(result_df, tile_name)

    return


def compute_shifts(target_folder, reference_folder, tiles):
  """
  Launch computation of shifts for each tile
  """
  for tile, tile_geom in tiles.items():
      print(f"Computing shifts for {tile}")
      start = time.time()
      polygon = Polygon(tile_geom)
      gdf = gpd.GeoDataFrame(geometry=[polygon], crs='EPSG:4326').to_crs('EPSG:32632')

      # Find grid cubes in the tile
      tile_cubes = find_cubes_aoi(data_folder=target_folder, geom=gdf)
      tile_cubes = tile_cubes.set_crs(32632)
      end = time.time()
      print('Looking for all files to process', end-start)

      # Launch coreg without applying shift, just saving shifts
      processed_names = []
      if os.path.exists(f"shift_results_{tile}.pkl"):
          df = pd.read_pickle(f"shift_results_{tile}.pkl")
          processed_names = df.drop_duplicates('name', keep='first').name.tolist()

      process_tile(
          file_list=tile_cubes.file.tolist(),
          processed_names=processed_names,
          target_folder=target_folder,
          reference_folder=reference_folder,
          tile_name=tile,
          batch_size=40 # number of timestamps to process in parallel for each file
      )

  return