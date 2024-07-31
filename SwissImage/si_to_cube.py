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
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import zarr
import datetime



def extract_minx_maxy(file):
    """ 
    Extract coordinates of topleft corner of zarr cube, as well as year
    """
    parts = file.split('_')
    minx = int(parts[1])
    maxy = int(parts[2])
    yr = int(parts[3][:4])
    return minx, maxy, yr

def check_processed_files(output_prefix, grid):
    """ 
    Verify which grid cells already have a corresponding file and don't need to be run
    """
    data_files = [f for f in os.listdir(output_prefix) if f.endswith('zarr')]
    df_zarr = pd.DataFrame(data_files, columns=['file'])
    
    if len(df_zarr):
        df_zarr[['minx', 'maxy', 'yr']] = df_zarr['file'].apply(lambda x: pd.Series(extract_minx_maxy(x)))
        grid['selected'] = grid['yr'].apply(lambda x: True if x else False)
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
        matching = [os.path.join(data_path,f) for f in all_files if file_pattern in f]
        if len(matching) > 1:
          raise Exception('There are multiple years of data available for a same location: adapt code to select year')
        else:
          files.append(matching[0])
        
    return files

def open_rasters(files):
    """ 
    Open rasters and add a variable year which provides acquistion year for each pixel

    :param files: list of paths of SwissImage TIFs to open
    :returns combined: xr.Dataset of all rasters RGB and year variables
    """
    years = [int(f.split('/')[-1].split('_')[1]) for f in files]
    rasters = []
    year_dataarrays = []

    for i, f in enumerate(files):
        raster = rioxarray.open_rasterio(f)
        rasters.append(raster)
        
        year_data = xr.DataArray(
            np.full_like(raster[0], years[i], dtype=np.int16),  # Match the shape of the raster
            dims=raster[0].dims,
            coords=raster[0].coords
        )
        year_dataarrays.append(year_data.expand_dims('band', axis=0))  # Ensure the same dimension

    # Combine rasters by coordinates
    combined = xr.combine_by_coords(rasters)
    combined_year = xr.combine_by_coords(year_dataarrays)
    combined = combined.to_dataset("band").rename({1:"R", 2:"G", 3:"B"})
    combined['year'] = combined_year[0]
    combined = combined.drop_vars(['spatial_ref', 'band'])

    return combined.to_array("band").rio.write_crs(2056)

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

def round_to_lower_even(number):
    rounded = round(number)
    if rounded % 2 == 0:
        return rounded
    else:
        return rounded - 1

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
    output_path = output_prefix + f'SwissImage{res}_{int(lon_min)}_{int(lat_max)}.zarr'

    # Save the patch to Zarr with compression
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    if overwrite or not os.path.exists(output_path):
        ds.to_zarr(output_path, consolidated=True, mode='w', encoding={var: {'compressor': compressor} for var in ds.data_vars})
        print('Saved patch', output_path) 
            
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
    start_index = 1000
    for i, row in grid.iloc[start_index:].iterrows(): #for i, row in grid.iterrows():
        if not grid_copy.loc[i, 'selected']:
            print(f"{datetime.datetime.now()}----Processing patch {i}/{len(grid)}----")
            
            # Find the SI images that would intersect with the grid cell
            patch_2056 = gpd.GeoDataFrame([row], crs=32632).to_crs(2056).geometry
            minx, miny, maxx, maxy = patch_2056.total_bounds
            all_files = [f for f in os.listdir(data_path)]
            files = match_si_to_patch(minx, miny, maxx, maxy, data_path, res, all_files)
            yrs = [int(f.split('/')[-1].split('_')[1]) for f in files]
            
            # Open and prepare SwissImage data
            da = open_rasters(files)

            # Shift coords to topleft of pixel
            da = coords_to_topleft(da, res)

            # Reproject to EPSG 32632
            da_reproj = da.rio.reproject("EPSG:32632", shape=(da.sizes['x'],da.sizes['y']), resampling=Resampling.cubic)
            # TO DO Resample to S2 grid
            start_x = round_to_lower_even(da_reproj['x'].values[0]) 
            end_x = round_to_higher_even(da_reproj['x'].values[-1])
            start_y = round_to_higher_even(da_reproj['y'].values[0])
            end_y = round_to_lower_even(da_reproj['y'].values[-1])
            new_x = np.arange(start_x, end_x+2, 2) # TO DO: then lengths should be fixed --> Actually should align to custom grid here!
            new_y = np.arange(start_y, end_y-2, -2)
            # TO DO: gets converted to float64 here
            da_resamp = da_reproj.interp(x = new_x, y = new_y, method = "cubic", kwargs={'fill_value':'extrapolate'})
          
            # Crop to cube bounds
            da_resamp = da_resamp.rio.clip_box(*row.geometry.bounds)
            # Convert to dataset
            ds = da_resamp.to_dataset("band") 
            ds["year"] = ds['year'].round().astype(int) # TO DO: check is this always correct?
            print(np.unique(yrs), np.unique(ds.year.values))
            # Update metadata
            ds.attrs = metadata
            # Save data
            #save_cube(ds, res, output_prefix, overwrite)

            


            
        if i==1005:
          break


    return grid_copy

if __name__ == "__main__":

    # Define data path
    data_path = os.path.expanduser('~/mnt/eo-nas1/data/swisstopo/raw/SwissImage_2m/')
    res = 2

    # Define output path
    output_prefix = os.path.expanduser('~/mnt/eo-nas1/data/swisstopo/cubes/SwissImage_2m/')
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
              "processing": "Rreprojected from EPSG:2056 to EPSG:32632. REsampled to closest even coordinates. Cubic resmapling used ad every step.",
            }

    grid_copy = run_processing(grid, grid_copy, output_prefix, data_path, res, metadata, overwrite)
    """ 
    any_false_selected = any(grid_copy['selected'] == False)
    if any_false_selected:
        print("There are False values in the 'selected' column.")
    else:
        print("All values in the 'selected' column are True.")
    """


