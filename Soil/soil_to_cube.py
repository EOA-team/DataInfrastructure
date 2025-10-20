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


def reproject_tif_old(arr, src_corners, dst_corners, src_crs=2056, dst_crs=32632, res=10):

    lef, bot, rig, top = src_corners
    lef_final, bot_final, rig_final, top_final = dst_corners

    transformer = Transformer.from_crs(dst_crs, src_crs, always_xy=True)
    xt = np.arange(lef_final, rig_final, res) 
    yt = np.arange(top_final, bot_final, -res) 
    # Transform just the axes for faster run
    x_new, _ = transformer.transform(xt, np.full_like(xt, yt[0]))
    _, y_new = transformer.transform(np.full_like(yt, xt[0]), yt)
    XT, YT = np.meshgrid(x_new, y_new, indexing="ij")
    
    # Interpolatle merged_array to new coords in src CRS
    x_old = np.arange(lef, rig, res)
    y_old = np.arange(top, bot, -res)
    f = RegularGridInterpolator((y_old,x_old), arr, method='linear',bounds_error=False, fill_value=np.nan)
    reproj_arr = f((YT, XT)).T.astype(np.float64)

    return reproj_arr


def reproject_tif(arr, src_corners, dst_corners, src_crs=2056, dst_crs=32632, xres=30, yres=-30, final_res=10, nodata=-99999):

    print(dst_corners)

    # Make a mask of valid data
    valid_mask = np.isfinite(arr) & (arr != nodata)
    arr = np.where(valid_mask, arr, 0)  # replace nodata with 0 for interpolation

    lef, bot, rig, top = src_corners
    lef_final, bot_final, rig_final, top_final = dst_corners

    transformer = Transformer.from_crs(dst_crs, src_crs, always_xy=True)
    xt = np.arange(lef_final, rig_final, final_res) 
    yt = np.arange(top_final, bot_final, -final_res) 

    # Transform just the axes for faster run
    x_new, _ = transformer.transform(xt, np.full_like(xt, yt[0]))
    _, y_new = transformer.transform(np.full_like(yt, xt[0]), yt)
    XT, YT = np.meshgrid(x_new, y_new, indexing="ij") # so XT will have shape (len(y_new), len(x_new))

    # Convert projected coordinates into array index space
    x_idx = (XT - x_new[0]) / np.abs(xres) # (XT - lef) / np.abs(final_res)
    y_idx = (top - YT) / np.abs(yres)
    print(x_idx)

    # Map_coordinates expects coords as (rows, cols)
    coords = np.array([y_idx.ravel(), x_idx.ravel()])

    # Interpolate both the image and the mask
    data_interp = map_coordinates(arr, coords, order=1, mode="constant", cval=0)
    mask_interp = map_coordinates(valid_mask.astype(float), coords, order=1, mode="constant", cval=0)

    # Normalize (avoid dividing by 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        reproj_arr = data_interp / mask_interp
    reproj_arr[mask_interp < 1e-6] = np.nan  # where no valid data contributed
  
    return reproj_arr.reshape(XT.shape).T 


def process_patch(i, row, ds, datavar, output_dir):
    """Process and save a single grid patch to Zarr."""

    patch = row.geometry
    minx, miny, maxx, maxy = patch.bounds
    output_path = output_dir + f'/cccsols_{int(minx)}_{int(maxy)}.zarr'

    ds = ds.sel(lon=slice(minx, maxx-10), lat=slice(maxy, miny+10))

    if not np.isnan(ds[datavar]).all():

      if os.path.exists(output_path):
        # Open ds and add variable
        ds_update = xr.open_zarr(output_path)
        ds_update[datavar] = ds[datavar]
      else:
        # Create new
        ds_update = ds.copy()

      # Update metadata
      attrs = ds_update.attrs
      if 'history' not in attrs.keys():
        attrs['history'] = f"Reprojected and regrid datacube to EPSG:32632 by Sélène Ledain on {date.today()}"
      ds_update.attrs = attrs
        
      # Chunk
      ds_update = ds_update.chunk({'lat': -1, 'lon': len(ds_update.lon)/2}) 
      
      # Save to Zarr with compression
      ds_update.to_zarr(output_path, consolidated=True, mode='w')
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



data_dir = os.path.expanduser('~/mnt/eo-nas1/data/soil/ccsols/Daten_2024-01') #os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/DataInfrastructure/Soil')
grid_path = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/gridface_s2tiles_CH.shp')
output_dir = os.path.expanduser('~/mnt/eo-nas1/data/soil/ccsols/datacubes')

grid = gpd.read_file(grid_path)


# Find extent of S2 grid

for f in os.listdir(data_dir): 
  if 'err' in f:
    continue

  datavar = f.split('Soil_')[1].split('.tif')[0] #'debug' #

  with rasterio.open(os.path.join(data_dir, f)) as src:
    array = src.read(1)        # first band as numpy array
    profile = src.profile      # metadata (dtype, crs, transform, etc.)
    crs = src.crs              # coordinate reference system
    transform = src.transform  # affine transform
    bounds = src.bounds
    xres, yres = src.res 

  valid_mask = np.isfinite(array) & (array != -99999)
  array = np.where(valid_mask, array, np.nan)

  #print(bounds)  # check if botteom < top, if not adjust order in array 
  bounds = np.array([bounds.left, bounds.bottom, bounds.right, bounds.top]) # [left, bottom, right, top]

  # --- Put data in xr.DataSet ---
  da = xr.DataArray(
      np.expand_dims(array, axis=0),
      dims=("band", "lat", "lon"),
      coords={
          "band": np.arange(1, 2, 1),
          "lat": np.arange(bounds[3], bounds[1], -yres),
          "lon": np.arange(bounds[0], bounds[2], xres),
      }
  )
  ds = da.to_dataset('band').rename({1:datavar})
  ds = ds.rio.write_crs(2056).rio.set_spatial_dims(x_dim='lon', y_dim='lat') 

  # --- Reference dataset (grid) ---
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
  ds_grid = da_grid.to_dataset('band').rename({1:datavar})
  ds_grid = ds_grid.rio.write_crs(32632).rio.set_spatial_dims(x_dim='lon', y_dim='lat') 

  # --- Reproject data ---
  ds = ds.rio.reproject_match(ds_grid).rename({'x':'lon', 'y':'lat'}) # will resample with nearest interp and reproject
  ds = ds.drop_vars('spatial_ref')
  """ 
  ds = ds.assign_coords({
      "lon": ds.lon + 5,
      "lat": ds.lat - 5
  })
  ds.rio.write_crs(32632).rename({'lon':'x', 'lat':'y'})[datavar].rio.to_raster('final.tif')
  """
  print(ds.compute())
  # Slice and save
  slice_and_save(ds, grid, datavar, output_dir)

  

