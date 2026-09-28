import pandas as pd
import os
import geopandas as gpd
from shapely import Polygon, box
import xarray as xr
import numpy as np
import glob
from collections import defaultdict
from scipy.ndimage import affine_transform
import datetime
import zarr
import warnings
warnings.filterwarnings('ignore')


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
  # 1. Filter S2 based on geom intersection
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
      for i in range(ds.sizes['time']):
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


def split_and_save(ds, minx, maxy, output_folder, attrs):
    """
    Extract cube of interest, update attributes and save data year by year

    :param ds: xr Dataset with coregistered data
    :param minx: min lon of cube
    :param maxy: max lat of cube
    :param output_folder: folder where to write new data
    :param attrs: original metadata of cube
    """
    if ds.lat.values[0] < ds.lat.values[-1]:
      ds = ds.isel(lat=slice(None, None, -1)) 
    ds = ds.sel(lon=slice(minx, minx+1270), lat=slice(maxy, maxy-1270))
    
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
          if 1: #not os.path.exists(output_path):
            print('Saving', output_path)
            ds_yr.to_zarr(output_path, consolidated=True, mode='w', encoding={var: {'compressor': compressor} for var in ds_yr.data_vars})
        
          
    return


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




if __name__ == "__main__":
  
  target_folder = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH')
  output_folder = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/coreg/CH')

  tiles = {'32TMT': [(7.66286529045287, 47.845914826827),(7.68758850048273, 46.8581759574933),(9.12805716115049, 46.8656282472998),(9.1304703619161, 47.8536277114283),(7.66286529045287, 47.845914826827)],
            '32TLT': [(6.32792675947954,47.8225951267953),(6.37728036218156, 46.8356437136561),(7.81664846692137, 46.8595830833696),(7.79435498889756, 47.8473711518664),(6.32792675947954, 47.8225951267953)],
            '32TNT': [(8.9997326423426, 47.8537018420053),(8.99973758744939,46.8656998728757),(10.4401498244026, 46.8566398814282),(10.4672773767603, 47.8443250460311),(8.9997326423426, 47.8537018420053)],
            '32TNS': [(8.99973715756222,46.9537091787344),(8.9997418700468, 45.9655511480415),(10.4166561015996, 45.9567698586004),(10.442508097938, 46.9446214409232),(8.99973715756222, 46.9537091787344)],
            '32TMS': [(7.68543924709582, 46.9461622205863),(7.7089998702566, 45.9582586884214),(9.12596726307512, 45.9654817261501),(9.12826694512377, 46.9536373337673),(7.68543924709582, 46.9461622205863)],
            '32TLS': [(6.37298980762373, 46.9235610080068),(6.42002507864337, 45.9364192287437),(7.83595551698163, 45.9596225320207),(7.81471043979331, 46.94757365544),(6.37298980762373, 46.9235610080068)],
            '31TGM': [(5.62648535526562, 46.9235730577955),(5.57945945554239, 45.9364308725672),(6.99300216652316, 45.8957352511058),(7.06565713267192, 46.8814596607493),(5.62648535526562, 46.9235730577955)],
            '31TGN': [(5.67153942638492, 47.8226075594756),(5.62219565549863, 46.8356557266896),(7.05902986502132, 46.793670687313),(7.13525847525188, 47.7791570713891),(5.67153942638492, 47.8226075594756)]
          }

  ### Step 1: compute avg shifts

  for tile, tile_geom in tiles.items():
    if os.path.exists(f"shift_results_{tile}.pkl"):
        df = pd.read_pickle(f"shift_results_{tile}.pkl")
    
    df = df.dropna(subset='uri')
    df['date'] = df['uri'].apply(lambda x: x.split('_')[2].split('T')[0])
    df['tile'] = df['uri'].apply(lambda x: x.split('_')[5])

    # Keep tile of interest
    df = df[df['tile'] == f'T{tile}']

    # Compute mean shift per uri (as there can be multiple uri on a same date)
    df = df.drop(['name', 'date', 'tile'], axis=1).groupby('uri').mean().reset_index()


    ### Step 2: find cubes in tile
    polygon = Polygon(tile_geom)
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs='EPSG:4326').to_crs('EPSG:32632')
      
    # Find grid cubes in tile 
    tile_cubes = find_cubes_aoi(data_folder=target_folder, geom=gdf)
    tile_cubes = tile_cubes.set_crs(32632).reset_index(drop=True)
    
 
    
    ### Step 3: apply shifts to a cube 
    i = tile_cubes[(tile_cubes['minx']==462300) & (tile_cubes['maxy']==5256620)].index.values[0]
    f = tile_cubes.file.values[i]
    ds, minx, maxy, attrs = load_cubes(f, target_folder)

    if ds.lat.values[0] < ds.lat.values[-1]:
      ds = ds.isel(lat=slice(None, None, -1)) 
    print(ds.sizes)
  
    lon_size, lat_size, time_size = ds.sizes['lon'], ds.sizes['lat'], ds.sizes['time']

    to_drop = ['mean_sensor_azimuth', 'mean_sensor_zenith', 'mean_solar_azimuth', 'mean_solar_zenith', 'product_uri']
    ds_tgt = ds.drop_vars(to_drop)
    bands = list(ds_tgt.data_vars)

    coreg_stack = []

    # Loop over time and apply shift where needed
    for i in range(ds.sizes['time']):
      t = ds.time.values[i]
      uri = ds.isel(lon=lon_size//2, lat=lat_size//2, time=i).product_uri.values
      if not df.loc[df.uri == uri].empty:
        shifts = (df.loc[df.uri == uri, 'shift_x'].values[0], df.loc[df.uri == uri, 'shift_y'].values[0])
      else:
        shifts = (None, None)  # or set a default value
      
      # Apply shifts
      if not any(v is None or np.isnan(v) for v in shifts):
        tgt_image = ds_tgt.isel(time=i).to_array().values.transpose(1, 2, 0) #lat, lon, band
        corrected_image = apply_shifts(tgt_image, [shifts[0], shifts[1]], tgt_image.shape)
        coreg_stack.append(corrected_image)
      else:
        coreg_stack.append(ds_tgt.isel(time=i).to_array().values.transpose(1, 2, 0))
      
    # Convert back to xr.DataSet format
    corrected_images_stack = np.stack(coreg_stack, axis=0).transpose((3,0,1,2))  # Final shape is (bands, time, lat, lon)

    # Create a new xarray Dataset
    time_dim = ds_tgt.sizes['time']
    lat_dim = ds_tgt.sizes['lat']
    lon_dim = ds_tgt.sizes['lon']
    bands_dim = len(bands)

    # Create DataArray for each band
    data_vars = {band: xr.DataArray(
        data=corrected_images_stack[bands.index(band),:, :, :],
        dims=['time','lat','lon'],
        coords={'lat': ds_tgt['lat'].values, 'lon': ds_tgt['lon'].values, 'time': ds_tgt['time'].values},
        name=band
    ) for band in bands}

    # Create Dataset
    ds_coreg = xr.Dataset(data_vars=data_vars)

    # Convert back to uint16
    ds_coreg = ds_coreg.fillna(65535).clip(0, 65535).round()
    ds_coreg = ds_coreg.astype(np.uint16)

    # Add back variables
    ds_coreg[to_drop] = ds[to_drop]
    print(ds_coreg.sizes)


    # Save central cube
    offsets = [-1, 0, 1]
    for dx in offsets:
      for dy in offsets:
          # Compute the bottom-left coordinates for each cube
          cube_x = minx + dx * 1280
          cube_y = maxy + dy * 1280
          split_and_save(ds_coreg, cube_x, cube_y, output_folder, attrs)


    break


  
   



  
