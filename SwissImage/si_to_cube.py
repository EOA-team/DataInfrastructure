"""
Load raw TIF files from SwissImage dataset and populate cubes from custom grid.

Sélène Ledain
selene.ledain@agroscope.admin.ch
Jul. 31 2024
"""

import os
import pandas as pd
import geopandas as gpd
import threading
import datetime
import glob
import rioxarray
import xarray as xr
from rasterio.enums import Resampling
import rasterio
from rasterio.vrt import WarpedVRT
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import zarr
import datetime
import math
import time
from affine import Affine
import time



def extract_minx_maxy(file):
    """ 
    Extract coordinates of topleft corner of zarr cube, as well as year
    """
    parts = file.split('_')
    minx = int(parts[1])
    maxy = int(parts[2].split('.zarr')[0])
    return minx, maxy

def check_processed_files(output_prefix, grid):
    """ 
    Verify which grid cells already have a corresponding file and don't need to be run
    """
    data_files = [f for f in os.listdir(output_prefix) if f.endswith('zarr')]
    df_zarr = pd.DataFrame(data_files, columns=['file'])
    
    if len(df_zarr):
        df_zarr[['minx', 'maxy']] = df_zarr['file'].apply(lambda x: pd.Series(extract_minx_maxy(x)))
        grid = grid.merge(df_zarr, how='left', right_on=['minx', 'maxy'], left_on=['left', 'top'])
        grid['selected'] = ~grid['file'].isna()
    else:
        grid['selected'] = [False]*len(grid)

    grid_copy = grid.copy()
    return grid_copy

def floor_to_nearest_thousand(value):
    return int((value // 1000) * 1000)

def match_si_to_patch(minx, miny, maxx, maxy, data_path, res, all_files):
    """ 
    Find SwissImage files that fall within bounds of a patch
    :params minx, miny, maxx, maxy: total bounds of patch in EPSG 2056
    :param data_path: where SwissImage raw TIF files are stored 
    :param all_files: all files in data_path
    :param res: resolution of SwissImage (0.1 or 2)
    """
    # Find the range of thousandths for x and y
    minx_rounded = floor_to_nearest_thousand(minx)
    maxx_rounded = floor_to_nearest_thousand(maxx)
    miny_rounded = floor_to_nearest_thousand(miny)
    maxy_rounded = floor_to_nearest_thousand(maxy)

    # Create lists of thousandths between min and max for x and y
    x_thousandths = list(range(minx_rounded // 1000, (maxx_rounded // 1000) + 1))
    y_thousandths = list(range(miny_rounded // 1000, (maxy_rounded // 1000) + 1))

    files = []
    for x in x_thousandths:
      for y in y_thousandths:
        # Find file in data_path
        file_pattern = f"_{x}-{y}_{res}_2056.tif"
        matching = [os.path.join(data_path,f) for f in all_files if file_pattern in f and f.endswith('tif')]
        if len(matching) > 1:
          raise Exception('There are multiple years of data available for a same location: adapt code to select year')
        if len(matching) == 0:
          continue
        else:
          files.append(matching[0])
        
    return files

def open_rasters(files):
    """ 
    Open rasters and add a variable year which provides acquistion year for each pixel

    :param files: list of paths of SwissImage TIFs to open
    :returns combined: xr.Dataset of all rasters RGB and year variables
    """
    rasters = []

    for i, f in enumerate(files):
        raster = rioxarray.open_rasterio(f)
        raster = raster.rio.write_nodata(255)
        rasters.append(raster)
        
    # Combine rasters by coordinates
    combined = xr.combine_by_coords(rasters, combine_attrs='override').fillna(255).rio.write_nodata(255)
    #combined = combined.to_dataset("band").rename({1:"R", 2:"G", 3:"B"})
    #combined = combined.drop_vars(['spatial_ref'])

    return combined#.to_array("band").rio.write_crs(2056)

def coords_to_topleft(ds, res):
  # Adjust coords to topleft corner of pixel
  x_coords = ds['x']
  y_coords = ds['y']

  ds = ds.assign_coords({'x': x_coords - res / 2, 'y': y_coords + res / 2})
  return ds

def round_to_higher_even(number):
    rounded = round(number)
    if rounded % 2 == 0:
        return rounded + 2
    else:
        return rounded + 1

def round_to_higher_dec(number):
    return math.ceil(number / 0.1) * 0.1

def round_to_lower_dec(number):
    return math.floor(number / 0.1) * 0.1

def round_to_lower_even(number):
    rounded = round(number)
    if rounded % 2 == 0:
        return rounded
    else:
        return rounded - 1

def resample_to_s2_old(da, res):
    """
    Resample so that coords fall every even value if res is 2m, or 0.1 if res is 0.1m

    :param da: xr DataArray
    :param res: 0.1 or 2
    :return da: resampled xr DataArray
    """
    if res == 2:
        start_x = round_to_lower_even(da['x'].values[0]) 
        end_x = round_to_higher_even(da['x'].values[-1])
        start_y = round_to_higher_even(da['y'].values[0])
        end_y = round_to_lower_even(da['y'].values[-1])
        new_x = np.arange(start_x, end_x+2, 2) # TO DO: then lengths should be fixed --> Actually should align to custom grid here!
        new_y = np.arange(start_y, end_y-2, -2)
    if res == 0.1:
      start_x = round_to_lower_dec(da['x'].values[0]) 
      end_x = round_to_higher_dec(da['x'].values[-1])
      start_y = round_to_higher_dec(da['y'].values[0])
      end_y = round_to_lower_dec(da['y'].values[-1])
      new_x = np.arange(start_x, end_x+0.1, 0.1)
      new_y = np.arange(start_y, end_y-0.1, -0.1)
    
    da = da.interp(x=new_x, y=new_y, method='cubic', kwargs={'fill_value': 'extrapolate'})
    da = da.round().astype(int).clip(min=0, max=255)

    """
    da_masked = da.where(da != 255, np.nan).rio.write_nodata(np.nan)
    da_resamp = da_masked.interp(x=new_x, y=new_y, method='cubic').round() #kwargs={'fill_value': 'extrapolate'}
    da_resamp = da_resamp.where(~np.isnan(da_resamp), 255).rio.write_nodata(255) #replace NaNs back to the no-data value if needed
    da_resamp = da_resamp.astype(int) 
    """
    return da

def resample_to_s2(da, res):
    """
    Resample so that coords fall every even value if res is 2m, or 0.1 if res is 0.1m

    :param da: xr DataArray
    :param res: 0.1 or 2
    :return da: resampled xr DataArray
    """
    if res == 2:
        start_x = round_to_lower_even(da['x'].values[0]) 
        end_x = round_to_higher_even(da['x'].values[-1])
        start_y = round_to_higher_even(da['y'].values[0])
        end_y = round_to_lower_even(da['y'].values[-1])
        new_x = np.arange(start_x, end_x+2, 2) # TO DO: then lengths should be fixed --> Actually should align to custom grid here!
        new_y = np.arange(start_y, end_y-2, -2)
    if res == 0.1:
      start_x = round_to_lower_dec(da['x'].values[0]) 
      end_x = round_to_higher_dec(da['x'].values[-1])
      start_y = round_to_higher_dec(da['y'].values[0])
      end_y = round_to_lower_dec(da['y'].values[-1])
      new_x = np.arange(start_x, end_x+0.1, 0.1)
      new_y = np.arange(start_y, end_y-0.1, -0.1)
    
    width = len(new_x)
    height = len(new_y)
    
    transform = rasterio.transform.from_bounds(start_x, end_y-res, end_x+res , start_y , width, height) 
    """ 
    start = time.time()
    # Adapt transform since rio.reproject thinks coords are center of pixel (add/remove res/2)
    da = da.rio.reproject(dst_crs=da.rio.crs, 
            transform=transform,
            shape=(height, width),
            resampling=Resampling.cubic
        )
    end = time.time()
    print('Resampling with rio reproj', end-start)
    """
    da = reproject_dataarray(da, res=res, output_shape=(width, height), resampling_method=Resampling.cubic, src_crs=da.rio.crs, dst_crs=da.rio.crs, transform=transform)
    da = da.round().astype(int).clip(min=0, max=255)

    return da

def reshape_data(row, da, res):
    """
    Reshape data to same as geometry, and fill missing values with 255
    """
    geometry_bounds = row.geometry.bounds
    template = xr.DataArray(
        np.full((int(1280/res), int(1280/res)), 255),  # Fill with no data value initially
        dims=["y", "x"],
        coords={
            "y": np.arange(geometry_bounds[3], geometry_bounds[1], -res),  # y from top to bottom
            "x": np.arange(geometry_bounds[0], geometry_bounds[2], res),   # x from left to right
        },
        name="template"
    ).rio.write_crs(32632)
    da_clipped_reprojected = da.rio.reproject_match(template, resampling=Resampling.cubic)#, **reproject_kwargs={'dst_crs':'EPSG:32632'})
    da_filled = da_clipped_reprojected.fillna(255)

    return da_filled

def save_cube(ds, res, output_prefix, overwrite):
    """
    Write xarray to compressed zarr
    :param ds: xarray Dataset
    :param res: resolution of SwissImage (0.1 or 2)
    :param output_prefix: fodler where data should be written
    :param overwrite: whether to overwrite existing file of same name
    """
    # Define the output path for the Zarr store
    lat_max = ds.y.max().values
    lon_min = ds.x.min().values
    output_path = os.path.join(output_prefix, f'SwissImage{res}_{int(lon_min)}_{int(lat_max)}.zarr')

    # Save the patch to Zarr with compression
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    if overwrite or not os.path.exists(output_path):
        ds.to_zarr(output_path, consolidated=True, mode='w', encoding={var: {'compressor': compressor} for var in ds.data_vars})
        print('Saved patch', output_path) 

def reproject_dataarray(da, res, output_shape, resampling_method, src_crs, dst_crs, transform=None):
    """
    Reproject an xarray DataArray using rasterio.vrt.WarpedVRT.

    :param da (xarray.DataArray): Input DataArray to be reprojected.
    :param res (int or float): resolution of data in meters.
    :param output_shape (tuple): The desired output shape as (width, height).
    :param resampling_method (rasterio.enums.Resampling): Resampling method to use.
    :param src_crs (str): Source CRS in EPSG format (e.g., "EPSG:4326").
    :param dst_crs (str): Target CRS in EPSG format (e.g., "EPSG:32632").
    :param transform (optional, Affine): The transform of the input data. If not provided, it will be calculated from the input data.

    Returns:
    xarray.DataArray: Reprojected DataArray.
    """
    width, height = output_shape
    #print(output_shape)

    # Calculate the new transform
    if transform is None:
        transform, _, _ = rasterio.warp.calculate_default_transform(
            src_crs, dst_crs, da.sizes['x'], da.sizes['y'], *(da.x.values[0], da.y.values[-1]-res, da.x.values[-1]+res, da.y.values[0])
        )

    # Write the DataArray to a temporary file
    input_path = f'/tmp/input_{datetime.datetime.now()}.tif'
    da.rio.to_raster(input_path)

    reprojected_data = {}
    with rasterio.open(input_path) as src:
        with WarpedVRT(src, crs=dst_crs, width=width, height=height, resampling=resampling_method, transform=transform) as vrt:
            for i in range(1, src.count + 1):
                band_data = vrt.read(i)
                #print(band_data.shape)
                # Create a new DataArray for the reprojected band
                reprojected_data[f'band_{i}'] = xr.DataArray(
                    band_data,
                    dims=['y', 'x'],
                    coords={'x': np.linspace(transform[2], transform[2] + transform[0] * (width - 1), width),
                            'y': np.linspace(transform[5], transform[5] + transform[4] * (height - 1), height)}
                )

    # Remove the temporary file
    os.remove(input_path)

    combined_data = xr.concat([reprojected_data[f'band_{i}'] for i in range(1, src.count + 1)], dim='band')
    combined_data.coords['band'] = np.arange(1, src.count + 1)
    combined_data = combined_data.rio.write_crs(dst_crs)
    combined_data = combined_data.rio.write_nodata(255)

    return combined_data

def run_processing(grid, grid_copy, output_prefix, data_path, res, metadata, overwrite):
    """
    Download data for all grid cells and save each to zarr year by year

    :param grid: original grid used for downloading
    :param grid_copy: copy of grid that will be updated to track what geometries and yeas have been downloaded
    :param output_prefix: path to save zarr stores
    :param data_path: path where raw data stored
    :param res: resolution of SwissImage (0.1 or 2)
    :param overwrite: if True, will overwrite existing zarr stores
    :param metadat: dict of metadata to add to zarr store
    return grid_copy: updated grid_copy
    """
    lock = threading.Lock()

    # Start download
    start_index = 400
    for i, row in grid.iloc[start_index:].iterrows(): #for i, row in grid.iterrows():
        all_files = [f for f in os.listdir(data_path)]
        if not grid_copy.loc[i, 'selected'] or overwrite is True:
            start_all = time.time()
            print(f"{datetime.datetime.now()}----Processing patch {i}/{len(grid)}----")
            
            # Find the SI images that would intersect with the grid cell
            patch_2056 = gpd.GeoDataFrame([row], crs=32632).to_crs(2056).geometry
            minx, miny, maxx, maxy = patch_2056.total_bounds
            files = match_si_to_patch(minx, miny, maxx, maxy, data_path, res, all_files)
            
            # Open and prepare SwissImage data
            if len(files):
                da = open_rasters(files)
            else:
                continue

            # Shift coords to topleft of pixel     
            da = coords_to_topleft(da, res)

            # Reproject to EPSG 32632
            da = reproject_dataarray(da, res=res, output_shape=(da.sizes['x'],da.sizes['y']), resampling_method=Resampling.cubic, src_crs=da.rio.crs, dst_crs="EPSG:32632")
            # da_reproj = da.rio.reproject("EPSG:32632", shape=(da.sizes['x'],da.sizes['y']), resampling=Resampling.cubic)

            # Check if data fills in the patch. If not, will need to reshape before resampling and clipping
            minx, miny, maxx, maxy = row.geometry.bounds
            if da.x.min().item() > minx or da.x.max().item() < maxx or da.y.min().item() > miny or da.y.max().item() < maxy:
                #print('Doesnt fill cell')
                da = reshape_data(row, da, res)
                da = resample_to_s2(da, res)
            else:
                #print('Fills cell')
                # Directly resample
                da = resample_to_s2(da, res)
                
            # Clip to grid cell
            da = da.rio.clip_box(*(row.geometry.bounds[0], row.geometry.bounds[1]+res, row.geometry.bounds[2]-res, row.geometry.bounds[3]))
            # Convert to dataset
            ds = da.to_dataset("band").rename({1:"R", 2:"G", 3:"B"}).drop_vars('spatial_ref').rio.write_crs(32632)
            # Update metadata
            source_files = [f.split('/')[-1] for f in files]
            metadata.update({'source_files': ', '.join(source_files)})
            ds.attrs = metadata
            # Save data
            save_cube(ds, res, output_prefix, overwrite)
            end_all = time.time()
            print('Took:', end_all-start_all)
        
            """
            transform = Affine(res, 0.0, row.geometry.bounds[0], 0.0, -res, row.geometry.bounds[3])
            ds.rio.write_transform(transform, inplace=True)
            print(ds.spatial_ref.GeoTransform)
            print(ds.rio.transform())
            ds[['R', 'G', 'B']].rio.to_raster(f'pipeline_test_{i}.tif')
            """


        break
    return grid_copy

if __name__ == "__main__":

    # Define data path
    data_path = os.path.expanduser('~/mnt/eo-nas1/data/swisstopo/SwissImage/raw/10cm/')
    res = 0.1 # always in meters

    # Define output path
    output_prefix = os.path.expanduser('~/mnt/eo-nas1/data/swisstopo/SwissImage/cubes/10cm/')
    overwrite = False # If True, will overwrite existing files of same name

    # Define path to grid
    grid_path = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/gridface_s2tiles_CH.shp')
    grid = gpd.read_file(grid_path)

    # Check what is already processed and update grid based on files
    grid_copy = check_processed_files(output_prefix, grid)

    # Run processing
    metadata = {
              "history": f"Created by Sélène Ledain on {datetime.datetime.now()}. See https://github.com/EOA-team/DataInfrastructure/",
              "source": "Swisstopo SwissImage",
              "grid": f"EPSG:32632. Coordinates are upper left corners of pixels",
              "missing data fill value": 255,
              "processing": "Rreprojected from EPSG:2056. Resampled to closest even coordinates. Cubic resampling used at every step.",
            }

    grid_copy = run_processing(grid, grid_copy, output_prefix, data_path, res, metadata, overwrite)
    """
    any_false_selected = any(grid_copy['selected'] == False)
    if any_false_selected:
        print("There are False values in the 'selected' column.")
    else:
        print("All values in the 'selected' column are True.")
    """


