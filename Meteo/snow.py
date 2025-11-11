import os
import numpy as np
import rasterio
import geopandas as gpd
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
import xarray as xr
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import date
from scipy.ndimage import map_coordinates
import rioxarray
import time
import dask


def process_patch(i, row, ds, datavar, output_dir):
    """Process and save a single grid patch to Zarr."""

    patch = row.geometry
    minx, miny, maxx, maxy = patch.bounds
    output_path = output_dir + f'/snowdepth_{int(minx)}_{int(maxy)}.zarr'

    ds = ds.sel(lon=slice(minx, maxx-10), lat=slice(maxy, miny+10))

    if os.path.exists(output_path):
        # Append new time slice
        ds.to_zarr(
            output_path,
            mode='a',
            append_dim='time',   
            consolidated=False
        )
    else:
        # Create new Zarr store
        ds.to_zarr(
            output_path,
            mode='w',
            consolidated=False
        )
  
    print(f'Saved patch {i} for {datavar}', output_path)


    return


def slice_and_save(ds, grid, datavar, output_dir):
    """
    Regrid the weather data to the grid and save the datacubes to zarr using multithreading.

    :param ds: xarray dataset
    :param grid: geopandas dataframe
    :param datavar: variable name
    :param output_dir: dir to save the zarr files
    """
    
    with ThreadPoolExecutor(max_workers=12) as executor:  # Define the number of threads here
        # Submit a separate task for each row in the grid
        futures = [executor.submit(process_patch, i, row, ds, datavar, output_dir) for i, row in grid.iterrows()]
       
        # Wait for all threads to complete
        for i, future in enumerate(futures):
            try:
                future.result()  # This will raise an exception if the thread failed
                #print(f'Completed processing grid patch {i+1}/{len(grid)}')
            except Exception as e:
                print(f'Error processing patch {i+1}/{len(grid)}: {e}')
    
    return



data_path = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/DataInfrastructure/Meteo/HSCLQMD_ch01h.swiss.lv95_WY_1962_2023.nc')
grid_path = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/gridface_s2tiles_CH.shp')
output_dir = os.path.expanduser('~/mnt/eo-nas1/data/meteo/snowdepth')

# Reproject and resample to 10m

# --- Reference dataset (grid) ---
grid = gpd.read_file(grid_path)
xgrid = np.arange(grid.total_bounds[0], grid.total_bounds[2], 10)
ygrid = np.arange(grid.total_bounds[3], grid.total_bounds[1], -10)
da_grid = xr.DataArray(
    np.zeros((1,len(ygrid),len(xgrid))),
    dims=("band", "lat", "lon"),
    coords={
        "band": np.arange(1, 2, 1),
        "lat": ygrid,
        "lon": xgrid,
    }
)
ds_grid = da_grid.to_dataset('band')
ds_grid = ds_grid.rio.write_crs(32632).rio.set_spatial_dims(x_dim='lon', y_dim='lat') 

# --- Snow dataset (after 2015) ---

ds_snow = xr.open_dataset(data_path)
ds_snow = ds_snow.where(ds_snow.time >= np.datetime64("2015-01-01"), drop=True)
ds_snow = ds_snow.rio.write_crs(2056).rio.set_spatial_dims(x_dim='E', y_dim='N') 

"""
ds_snow = ds_snow.assign_coords({
    "E": ds_snow.E + 500,
    "N": ds_snow.N - 500
})
ds_snow.isel(time=100).rename({'N':'y', 'E':'x'}).rio.write_crs(2056).rio.to_raster('snow.tif')
"""

# --- Reproject using multiprocessing ---


def reproject_one_time(args):
    #da_t = ds.sel(time=t)
    da_t, t = args
    da_reproj = da_t.rio.reproject_match(ds_grid)
    return da_reproj.assign_coords(time=t).rename({'x':'lon', 'y':'lat'})


batch_size = 4
for i in np.arange(3142, len(ds_snow.time.values), batch_size): #3165 timestamps in total
   
    results = []
    time_vals = ds_snow.time.values[i:i+batch_size]
    args_list = [(ds_snow.sel(time=t), t) for t in time_vals]
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        batch_results = list(executor.map(reproject_one_time, args_list)) 
    results += batch_results

    ds_reproj = xr.concat(results, dim='time').drop_vars('swiss_lv95_coordinates')
    # Rename variable
    ds_reproj = ds_reproj.rename({'HSCLQMD':'depth'})
    # Drop attributes to avoid conflict
    ds_reproj['depth'] = ds_reproj['depth'].drop_attrs()

    slice_and_save(ds_reproj, grid, 'depth', output_dir)
 

"""
ds_reproj = ds_reproj.assign_coords({
    "x": ds_reproj.x + 5,
    "y": ds_reproj.y - 5
})
ds_reproj.isel(time=0).rio.write_crs(32632).rio.to_raster('snow_reproj.tif')
"""
