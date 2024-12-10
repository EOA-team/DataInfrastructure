import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
from scipy.interpolate import interp2d
import geopandas as gpd
import os
import re
import rasterio
#import shapefile
#from osgeo.gdal import GetDriverByName, GetDataTypeByName
#from osgeo import osr
#import geopandas
import json
import requests
from rasterio.warp import reproject, calculate_default_transform
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from shapely import Polygon
#from shapefile import Reader
#import matplotlib.path as mplp
#import time
import concurrent.futures
import threading
import zarr
import pandas as pd
from tqdm import tqdm
import multiprocessing
from functools import partial
from multiprocessing import shared_memory
import rioxarray
import xarray as xr
from rasterio.coords import BoundingBox
from datetime import date
import zarr
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator


def download_SA3D_STAC(
        bbox: Polygon, 
        out_crs: str, 
        out_res: float,
        server_url: str =  'https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d',
        product_res_label: str = '2'
    ):
    """
    DOWNLOAD_SA3D_STAC downloads the SwissAlti3D product for the bounding box 
    of a given shapefile and output resolution. The projection grid will start 
    from the lower and left box bounds. 
    
    Parameters
    ----------
    bbox_path : str
        path to the input shapefile, it can be a .gpkg or .shp with georef files
        in any crs
    out_crs : int
        epgs code of the output CRS (e.g. 4326)
    out_res : float
        output resolution
    server_url : str, optional
       Swisstopo STAC server url for the product.
       The default is 'https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d'.
    product_res_label : str, optional
        the original product comes in 0.5 or 2-m resolution. 
        The default is '2'.

    Returns
    -------
    xarray Dataset of buffered and reporjected data

    Example
    -------
    download_SA3D_STAC(
            bbox_path = bbox_fname, # bbox in any crs
            out_crs = 2056, # wanted output crs EPGS code
            out_res = 10, # wanted output resolution (compatible with out crs)
            server_url = 'https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d',
            product_res_label = '2' # can only be 0.5 or 2, original product resolution 
        )

    """

    lef_final, bot_final, rig_final, top_final = bbox.total_bounds

    # RETRIEVE TILE LINKS FOR DOWNLOAD 
    
    shp = bbox.to_crs('epsg:4326') #gpd.read_file(bbox_path).to_crs('epsg:4326') # WG84 necessary to the query, data in 2056
    lef = np.min(shp.bounds.minx)
    rig = np.max(shp.bounds.maxx)
    bot = np.min(shp.bounds.miny)
    top = np.max(shp.bounds.maxy)
    
    # If bbox is big divide the bbox in a series of chunks to override server limtiations
    cell_size = 0.05 #temporary bounding size in degree to define the query chunks 
    xbb = np.arange(lef,rig,cell_size)
    if xbb[-1] < rig:
        xbb = np.append(xbb,rig)
    ybb = np.arange(bot,top,cell_size)
    if ybb[-1] < top:
        ybb = np.append(ybb,top)
    
    files = []
    for i in range(len(xbb)-1):
        for j in range(len(ybb)-1):
            bbox_tmp = [xbb[i],ybb[j],xbb[i+1],ybb[j+1]]
            # construct bbox specification required by STAC
            bbox_expr = f'{bbox_tmp[0]},{bbox_tmp[1]},{bbox_tmp[2]},{bbox_tmp[3]}'
            # construct API GET call
            url = server_url + '/items?bbox=' + bbox_expr
            # send the request and check response code
            res_get = requests.get(url)
            res_get.raise_for_status()         
            # get content and extract the tile URLs
            content = json.loads(res_get.content)
            features = content['features']
            for feature in features:
                assets = feature['assets']
                tif_pattern = re.compile(r"^.*\.(tif)$")
                tif_files = [tif_pattern.match(key) for key in assets.keys()]
                tif_files =[x for x in tif_files if x is not None]
                tif_file = [x.string if x.string.find(f'_{product_res_label}_') > 0 \
                            else None for x in tif_files]
                tif_file = [x for x in tif_file if x is not None][0]
                # get download link
                link = assets[tif_file]['href']
                files.append(link)
    
    
    file_handler = []
    for row in files:
        src = rasterio.open(row,'r')
        file_handler.append(src)

    if len(file_handler):
        total_bounds = file_handler[0].bounds
        for src in file_handler[1:]:
            total_bounds = BoundingBox(
                left=min(total_bounds.left, src.bounds.left),
                bottom=min(total_bounds.bottom, src.bounds.bottom),
                right=max(total_bounds.right, src.bounds.right),
                top=max(total_bounds.top, src.bounds.top),
            )
        
        lef = total_bounds.left
        rig = total_bounds.right
        bot = total_bounds.bottom
        top = total_bounds.top
        
        merged_array, merged_transform = merge(datasets=file_handler, # list of dataset objects opened in 'r' mode
        bounds=(lef, bot, rig, top), # tuple
        nodata=65535, # float
        dtype='uint16', # dtype
        res=out_res,
        resampling=Resampling.nearest,
        method='first', # strategy to combine overlapping rasters
        )

        # Get the updated grid of the cube based in EPSG:2056
        transformer = Transformer.from_crs(32632, 2056, always_xy=True)
        xt = np.arange(lef_final, rig_final, 2) 
        yt = np.arange(top_final, bot_final, -2) 
        XT, YT = np.meshgrid(xt, yt, indexing='ij') 
        XT,YT = transformer.transform(XT,YT)
        
        # Interpolatle merged_array to new coords in EPSG:2056
        x_old = np.arange(lef, rig, 2)
        y_old = np.arange(top, bot, -2)
        f = RegularGridInterpolator((y_old,x_old),merged_array[0], method='linear',bounds_error=False, fill_value=0)
        merged_array_int = f((YT, XT)).T.astype(np.float64)


        da = xr.DataArray(
            np.expand_dims(merged_array_int, axis=0),
            dims=("band", "y", "x"),
            coords={
                "band": np.arange(1, 2, 1),
                "y": np.arange(top_final, bot_final, -2),
                "x": np.arange(lef_final, rig_final, 2),
            },
            attrs={
                "crs": str(32632),
                "transform": rasterio.transform.from_origin(lef_final, top_final, 2, -2),
                "nodata": 65535,
            },
        )
        ds = da.to_dataset('band').rename({1:'height'})
        #ds.rio.write_crs(32632).rio.to_raster('test_inverse.tif')

        return ds
        """
  
        # Close the input raster files
        #for fh in file_handler:
        #    fh.close()

        # Create a memory file for the merged raster
        with MemoryFile() as memfile:
            with memfile.open(
                driver='GTiff',
                height=merged_array.shape[1],
                width=merged_array.shape[2],
                count=merged_array.shape[0],
                dtype='uint16',
                crs=file_handler[0].crs,
                transform=merged_transform,
                nodata=65535
            ) as merged_raster:
                merged_raster.write(merged_array)
                
                # Reproject the merged raster
                #reprojected_raster = reproject_raster(merged_raster, out_crs)
                reprojected_da = reproject_raster_to_xarray(merged_raster, out_crs)
                reprojected_ds = reprojected_da.to_dataset('band').rename({1:'height'})
                
                return reprojected_ds
        """
    else:
      return None
    
        


if __name__ == "__main__":

    
  #####################
  # DEFINE PATHS AND VARIABLES

  grid_path = '~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/gridface_s2tiles_CH.shp'
  grid = gpd.read_file(grid_path)

  output_folder = os.path.expanduser('~/mnt/eo-nas1/data/swisstopo/DEM_new/')
  compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)

 
  #####################
  # DOWNLOAD FILES


  for i in range(len(grid)):
    cube = grid.iloc[[i]]
    minx, maxx, miny, maxy = cube.left.values[0], cube.right.values[0], cube.bottom.values[0], cube.top.values[0]
    output_path = os.path.join(output_folder, f'sa3d_{int(minx)}_{int(maxy)}.zarr')
    
    if not os.path.exists(output_path):
        
        ds = download_SA3D_STAC(bbox=cube, out_crs=32632, out_res=2)
        
        if ds is not None:
            """
            #ds.rio.write_crs(32632).rio.to_raster('test_code_cubic.tif')
            arr = ds.height.values
            arr[arr==65535] = 0
            plt.imshow(arr)
            plt.savefig('reproj_inv.png')

            arr = np.diff(arr)
            plt.imshow(arr)
            plt.savefig('diff_inv.png')
            """
           
            # Regrid and crop
            minx, maxx, miny, maxy = cube.left.values[0], cube.right.values[0], cube.bottom.values[0], cube.top.values[0]
            
            # Save to zarr
            attrs = ds.attrs
            attrs.pop('transform')
            attrs['history'] = f"Downloaded from swisstalti3D (swisstopo) on {date.today()}. Reprojected and regrid datacube to EPSG 32632 by Sélène Ledain."
            ds.attrs = attrs
            
            output_path = os.path.join(output_folder, f'sa3d_{int(minx)}_{int(maxy)}.zarr')
            ds.to_zarr(output_path, consolidated=True, mode='w', encoding={var: {'compressor': compressor} for var in ds.data_vars})
            print(f'Saved patch {i}', output_path, ds.dims)

