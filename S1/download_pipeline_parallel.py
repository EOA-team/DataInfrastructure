import os
from pathlib import Path
import sys
import argparse

base_dir = Path(os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure'))
sys.path.insert(0, str(base_dir))
import earthnet_minicuber as emc

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
import numpy as np
import zarr
import datetime
import concurrent.futures
import threading
import gc
import shutil
import traceback

# Zarr group per orbit direction: open with xr.open_zarr(path, group="asc")
ORBIT_GROUPS = {"asc": "ascending", "desc": "descending"}

class ProgressLog:
    """
    Append-only CSV log of finished patch-years, so the output folder never has to be listed.
    A patch-year is only logged after its zarr store is completely written: a store without
    a log entry is incomplete and gets rewritten on the next run.
    Patch-years without any data are logged with n_asc = n_desc = 0 and no file.
    """
    columns = ['minx', 'maxy', 'year', 'n_asc', 'n_desc', 'file', 'finished']

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        if not os.path.exists(path):
            with open(path, 'w') as f:
                f.write(','.join(self.columns) + '\n')
        df = pd.read_csv(path)
        self.done = set(zip(df.minx.astype(int), df.maxy.astype(int), df.year.astype(int)))

    def is_done(self, minx, maxy, year):
        return (int(minx), int(maxy), int(year)) in self.done

    def add(self, minx, maxy, year, n_asc, n_desc, file):
        with self.lock:
            with open(self.path, 'a') as f:
                f.write(f"{int(minx)},{int(maxy)},{int(year)},{n_asc},{n_desc},{file},{datetime.datetime.now().isoformat(timespec='seconds')}\n")
                f.flush()
                os.fsync(f.fileno())
            self.done.add((int(minx), int(maxy), int(year)))

def parse_years(values):
    """
    Parse --years arguments: single years ("2022") and inclusive ranges ("2014-2025")

    :param values: list of strings
    :return: sorted list of int years
    """
    years = set()
    for value in values:
        if '-' in value:
            start, end = value.split('-')
            years.update(range(int(start), int(end) + 1))
        else:
            years.add(int(value))
    return sorted(years)

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

def save_cube(cube, n_cells, output_prefix, year, progress, patch_size=128, resolution=10, overwrite=False):
    """
    Take a xarray, slice into patches of 128x128 pixels, compress and save to zarr store.
    Each zarr store has one group per orbit direction ("asc", "desc"), each with its own time axis.
    
    :param cube: xarray, containing a year of Sentinel-1 data
    :param n_cells: n_cells^2 is the number of patches of 128x128 in the xarray
    :param output_prefix: path to save the zarr store
    :param year: year of the data, used for the progress log
    :param progress: ProgressLog, finished patch-years are skipped and new ones are logged
    :param patch_size: size of the patch in pixels
    :param resolution: resolution of the data in meters
    :param overwrite: If True, will overwrite existing zarr stores
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

            if progress.is_done(lon_start, lat_start, year) and not overwrite:
                continue
            
            # Define the output path for the Zarr store
            file = f'S1_{int(lon_start)}_{int(lat_start)}_{year_start}{month_start}{day_start}_{year_end}{month_end}{day_end}.zarr'
            output_path = output_prefix + file
            # Not in the progress log: leftover from an interrupted run, or overwrite requested
            if os.path.exists(output_path):
                shutil.rmtree(output_path)

            # Slice the cube
            patch_cube = cube.sel(lat=slice(lat_start, lat_end), lon=slice(lon_start, lon_end))

            n_times = {}
            for group, orbit_state in ORBIT_GROUPS.items():
                group_cube = patch_cube.isel(time=(patch_cube.orbit_state.values == orbit_state))

                # Drop dates without any valid pixel: the pass did not cover this patch
                has_data = (group_cube.s1_valid.sum(dim=['lat', 'lon']) > 0).values
                group_cube = group_cube.isel(time=has_data)
                n_times[group] = len(group_cube.time)
                if n_times[group] == 0:
                    continue

                group_cube = group_cube.chunk({'time': -1, 'lat': -1, 'lon': len(group_cube.lon)//2})
                group_cube.to_zarr(output_path, group=group, consolidated=True, mode='w', encoding={var: {'compressor': compressor} for var in group_cube.data_vars})
                print('Saved patch', output_path, group)

            progress.add(lon_start, lat_start, year, n_times['asc'], n_times['desc'], file if sum(n_times.values()) else '')
    return

def download_year(year, specs, n_cells, output_prefix, overwrite, mega_patch, progress):
    """
    Download data for a specific year and save to zarr

    :param year: Year to download data for
    :param specs: Dictionary with download specifications
    :param n_cells: Number of cells in the patch
    :param output_prefix: Path to save zarr stores
    :param overwrite: If True, will overwrite existing zarr stores
    :param mega_patch: geometries to download
    :param progress: ProgressLog of finished patch-years
    """
    try:
    
        print(f"{datetime.datetime.now()}: Downloading year {year}")
        specs = dict(specs, time_interval=f"{year}-01-01/{year}-12-31")
        # Call minicuber
        cube = emc.load_minicube(specs, compute=True, verbose=True)
        if cube is None:
            print(f"No Sentinel-1 data found for year {year}")
            for left, top in zip(mega_patch['left'], mega_patch['top']):
                if not progress.is_done(left, top, year):
                    progress.add(left, top, year, 0, 0, '')
            return year, True
        
        # Call a function to rechunk, slice data based on mega-patch, compress, save to zarr
        save_cube(cube, n_cells, output_prefix=output_prefix, year=year, progress=progress, overwrite=overwrite)
        return year, True
 
    except Exception as e:
        print(f"An error occurred while downloading data for year {year}: {e}")
        traceback.print_exc()
        return year, False

def run_download(grid, years, num_cells, patch_size, output_prefix, overwrite, specs, progress, max_retries=3):
        """
        Download data for all grid cells and save each to zarr year by year

        :param grid: grid used for downloading, with a 'selected' column marking cells that are done
        :param years: list of years to download
        :param num_cells: max number of cells to add to grid cell for download
        :param patch_size: size of patch in meters
        :param output_prefix: path to save zarr stores
        :param overwrite: if True, will overwrite existing zarr stores
        :param specs: dictionary with download specifications
        :param progress: ProgressLog of finished patch-years
        :param max_retries: number of attempts per year before giving up on it for this patch
        :return grid: updated grid
        """

        # Start download
        for i, row in grid.iterrows(): 
            if not grid.loc[i, 'selected']:
                print(f"{datetime.datetime.now()}----Downloading patch {i}/{len(grid)}----")
                
                # Add surrounding patches to create up to num_cells x num_cells mega-patch (use patch as upper left corner)
                patch = row.geometry
                n_cells, mega_patch = create_max_square(patch, grid, num_cells, patch_size, epsg=4326)
                print(f"Adding {n_cells**2 -1} patches to download. Cube has side {int(patch_size*(n_cells)/specs['resolution'])}")
                print(patch)

                # Update specs 
                specs["lon_lat"] = (patch.bounds[0], patch.bounds[-1]) # upper left corner
                specs["xy_shape"] = (int(patch_size*(n_cells)/specs["resolution"]), int(patch_size*(n_cells)/specs["resolution"]))
                
                # Only years not yet done for this patch, retry failed years up to max_retries times
                years_to_download = [year for year in years if overwrite or not progress.is_done(row['left'], row['top'], year)]
                attempt = 0

                while years_to_download and attempt < max_retries:
                    attempt += 1
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futures = [executor.submit(download_year, year, specs, n_cells, output_prefix, overwrite, mega_patch, progress) for year in years_to_download]
                        results = [future.result() for future in concurrent.futures.as_completed(futures)]
                        
                    years_to_download = [year for year, success in results if not success]

                if years_to_download:
                    print(f"Giving up on years {sorted(years_to_download)} for patch {i} after {max_retries} attempts. Rerun the script to retry them.")

                # Mark the selected cells
                grid.loc[mega_patch.index, 'selected'] = True

                # Clean up variables after each iteration
                del mega_patch, years_to_download
                gc.collect()  # Explicitly call garbage collector to free up memory

        return grid


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download Sentinel-1 RTC cubes for the Swiss grid")
    parser.add_argument('--years', nargs='+', default=['2014-2025'],
                        help="Years to download, e.g. --years 2022, --years 2016 2017, --years 2014-2025 (default: full archive 2014-2025, Planetary Computer RTC starts Oct 2014)")
    args = parser.parse_args()

    # Define download parameters
    YEARS = parse_years(args.years)
    patch_size = 1280 # meters
    num_cells = 8
    output_prefix = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel1/raw/CH/')
    overwrite = False # If True, will overwrite existing files of same name
    max_retries = 3 # attempts per year and patch before moving on
    os.makedirs(output_prefix, exist_ok=True)
    print(f"Downloading years {YEARS}")

    # Define path to grid
    grid_path = '~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/gridface_s2tiles_CH.shp'
    grid = gpd.read_file(grid_path)

    # Check what is already downloaded from the progress log (no listing of the output folder)
    progress = ProgressLog(output_prefix + 'progress_S1.csv')
    grid['selected'] = [all(progress.is_done(left, top, year) for year in YEARS) for left, top in zip(grid['left'], grid['top'])]
    print(f"{grid['selected'].sum()}/{len(grid)} grid cells already done for these years")
    
    specs = {
        "lon_lat": (None, None), # topleft
        "xy_shape": (None, None), # width, height of cutout around center pixel
        "resolution": 10, # in meters.. will use this on a local UTM grid..
        "time_interval": None, # set per year in download_year
        "final_epsg": 32632,
        "providers": [
            {
                "name": "s1",
                "kwargs": {
                    "bands": ["vv", "vh"],
                    "data_source": "planetary_computer"}
            }
            ]
    }

    
    grid = run_download(grid, YEARS, num_cells, patch_size, output_prefix, overwrite, specs, progress, max_retries)

    # Check that all patches were treated
    any_false_selected = any(grid['selected'] == False)
    
    if any_false_selected:
        print("There are False values in the 'selected' column.")
    else:
        print("All values in the 'selected' column are True.")
