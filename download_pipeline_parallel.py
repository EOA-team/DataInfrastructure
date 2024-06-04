import os
from pathlib import Path
import sys

base_dir = Path(os.path.dirname(os.path.realpath("__file__"))).parent
sys.path.insert(0, str(base_dir))
import earthnet_minicuber as emc

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.ops import cascaded_union
import matplotlib.pyplot as plt
import contextily as cx
import numpy as np
import zarr
import pickle
import time
import datetime
import concurrent.futures



def create_max_square(patch, grid, num_cells, patch_size, epsg=4326):
    """
    Use patch as upper left corner, and create biggest possible square.
    Max side length possible is num_cells*patch size.
    The returned mega-patch is a geopandas dataframe with a Polygon geometry. 
    The exterior coordinates of the square are given
    

    :param patch: Polygon (upper left corner polygon of square to create)
    :param grid: geodataframe with other polygons that can be used
    :param num_cells: max number of cells
    :param patch_size: side of a single patch
    :param epsg: coordinate system that the mega-patch should be returned in
    :return: n_cells which is the max number of cells, max_square which is a gdf containing the geometries added
    """

    n_cells = 0

    x, y = patch.exterior.coords.xy
    x_min, x_max, y_min, y_max = min(x), max(x), min(y), max(y)
    max_square = patch

    while n_cells <= num_cells: # Start with only patch, then adding cells
        # Calculate the coordinates of the square
        square = [
            (x_min, y_max),
            (x_max + patch_size * (n_cells), y_max),
            (x_max + patch_size * (n_cells), y_min - patch_size * (n_cells)),
            (x_min, y_min - patch_size * (n_cells)),
            (x_min, y_max)  # Close the polygon by repeating the first point
        ]
        
        # Check if this square is possible 
        square = Polygon(square)
        square_in_grid = grid[(grid.geometry.within(square)) & (~grid['selected'])]
        
        if len(square_in_grid) < (n_cells+1)**2: # the square has side ncells+1
            #print('Max found side', n_cells**2)
            return n_cells, max_square 
        else:
            max_square = square_in_grid
            n_cells += 1
            
    return num_cells, max_square

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

def save_cube(cube, n_cells, output_prefix, patch_size=128, resolution=10, overwrite=False):
    """
    Take a xarrays, slice into patches of 128x128 pixels, compress and save to zarr store
    
    :param mc: xarray, containing a year of Sentinel-2 data
    :param n_cells: n_cells^2 is the number of patches of 128x128 in the xarray
    :param output_prefix: path to save the zarr store
    :param patch_size: size of the patch in pixels
    :param resolution: resolution of the data in meters
    """

    # Find upper left corner of cube
    lat_max = cube.lat.max().values
    lon_min = cube.lon.min().values

    # Extract the start and end dates
    time_min, time_max = cube.time.min().values, cube.time.max().values

    year_start, month_start, day_start = extract_date(time_min)
    year_end, month_end, day_end = extract_date(time_max)
    
    # Create compressor
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    
    # Iterate over each patch
    for i in range(n_cells):
        for j in range(n_cells):
            lat_start = lat_max - i * patch_size * resolution
            lat_end = lat_start - (patch_size-1) * resolution
            lon_start = lon_min + j * patch_size * resolution
            lon_end = lon_start + (patch_size-1) * resolution
            
            # Slice the cube
            patch_cube = cube.sel(lat=slice(lat_start, lat_end), lon=slice(lon_start, lon_end))
            patch_cube = patch_cube.chunk({'time': -1, 'lat': -1, 'lon': -1}) # len(patch_cube.lon)/2
            
            # Define the output path for the Zarr store
            output_path = output_prefix + f'S2_{int(lon_start)}_{int(lat_start)}_{year_start}{month_start}{day_start}_{year_end}{month_end}{day_end}.zarr'

            # Save the patch to Zarr with compression
            if overwrite or not os.path.exists(output_path):
                patch_cube.to_zarr(output_path, consolidated=True, mode='w', encoding={var: {'compressor': compressor} for var in patch_cube.data_vars})
                print('Saved patch', output_path) # save_end-save_start
    return

def download_year(year, specs, n_cells, output_prefix, overwrite, mega_patch):
    """
    Download data for a specific year and save to zarr

    :param year: Year to download data for
    :param specs: Dictionary with download specifications
    :param n_cells: Number of cells in the patch
    :param output_prefix: Path to save zarr stores
    :param overwrite: If True, will overwrite existing zarr stores
    :param mega_patch: geometries to download
    """

    if grid_copy.loc[mega_patch.index, 'years_done'].isnull().any() or \
            not any(year in sublist for sublist in grid_copy.loc[mega_patch.index, 'years_done']):
        
        print(f"{datetime.datetime.now()}: Downloading year {year}")
        specs["time_interval"] = f"{year}-01-01/{year}-12-31"
        # Call minicuber
        cube = emc.load_minicube(specs, compute=True, verbose=True)
        
        # Call a function to rechunk, slice data based on mega-patch, compress, save to zarr
        save_cube(cube, n_cells, output_prefix=output_prefix, overwrite=overwrite)

    return year

def run_download(grid, grid_copy, num_cells, patch_size, output_prefix, overwrite, specs):
        """
        Download data for all grid cells and save each to zarr year by year

        :param grid: original grid used for downloading
        :param grid_copy: copy of grid that will be updated to track what geometries and yeas have been downloaded
        :param num_cells: max number of cells to add to grid cell for download
        :param patch_size: size of patch in meters
        :param output_prefix: path to save zarr stores
        :param overwrite: if True, will overwrite existing zarr stores
        :param specs: dictionary with download specifications
        :return grid_copy: updated grid_copy
        """

        # Start download
        for i, row in grid.iterrows():
            if not grid_copy.loc[i, 'selected']:
                print(f"{datetime.datetime.now()}----Downloading patch {i}/{len(grid)}----")
                
                # Add surrounding patches to create up to num_cells x num_cells mega-patch (use patch as upper left corner)
                patch = row.geometry
                n_cells, mega_patch = create_max_square(patch, grid_copy, num_cells, patch_size, epsg=4326)
                print(f"Adding {n_cells**2 -1} patches to download")

                # Update specs 
                specs["lon_lat"] = (patch.bounds[0], patch.bounds[-1]) # upper left corner
                specs["xy_shape"] = (int(patch_size*(n_cells+1)/specs["resolution"]), int(patch_size*(n_cells+1)/specs["resolution"]))
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(download_year, year, specs, n_cells, output_prefix, overwrite, mega_patch) for year in range(2017, 2024)]

                    successful_years = []
                    for future in concurrent.futures.as_completed(futures):
                        year = future.result() 
                        if year:
                            successful_years.append(year)

                    # Save the years already queried
                    grid_copy.loc[mega_patch.index, 'years_done'] = grid_copy.loc[mega_patch.index, 'years_done'].apply(\
                        lambda x: successful_years if x is None else x + successful_years)
                    grid_copy.to_pickle(output_prefix + 'grid_copy.pkl')
            
                # Mark the selected cells
                grid_copy.loc[mega_patch.index, 'selected'] = True
                grid_copy.to_pickle(output_prefix + 'grid_copy.pkl')


        return grid_copy


if __name__ == "__main__":

    # Define download parameters
    patch_size = 1280 # meters
    num_cells = 5
    output_prefix = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/')
    overwrite = False # If True, will overwrite existing files of same name


    # Define path to grid 
    grid_path = os.path.expanduser(output_prefix) + 'grid_copy.pkl'
    grid = pd.read_pickle(grid_path)
    #grid_path = '~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/gridface_s2tiles_CH.shp'
    #grid = gpd.read_file(grid_path)
    if 'selected' not in grid.columns:
        grid['selected'] = [False]*len(grid)
    if 'years_done' not in grid.columns:
        grid['years_done'] = [None]*len(grid)
    grid_copy = grid.copy()

    
    
    specs = {
        "lon_lat": (None, None), # center pixel
        "xy_shape": (None, None), # width, height of cutout around center pixel
        "resolution": 10, # in meters.. will use this on a local UTM grid..
        "time_interval": "2021-01-01/2021-03-31",
        "final_epsg": 32632,
        "providers": [
            {
                "name": "s2",
                "kwargs": {
                    "bands": ["AOT", "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "WVP"], 
                    "brdf_correction": True, 
                    "cloud_mask": False, 
                    "data_source": "planetary_computer"}
            }
            ]
    }

    
    grid_copy = run_download(grid, grid_copy, num_cells, patch_size, output_prefix, overwrite, specs)
        

    # Check that all patches were treated
    any_false_selected = any(grid_copy['selected'] == False)
    
    if any_false_selected:
        print("There are False values in the 'selected' column.")
    else:
        print("All values in the 'selected' column are True.")


