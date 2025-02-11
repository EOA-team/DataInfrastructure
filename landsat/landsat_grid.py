"""
Create a grid for Landsat cubes
Cube size is 128 x 128 landsat 30-m pixels = 3840 m
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
import rasterio
from rasterio.transform import from_origin
import os



# Create grid from coordinate, grid size and extent of raster to cover

landsat_tile_195_27 = (304485.0 - 15, 5375115.0 - 15)  # (ULX, ULY)
grid_size = 3840
meteo_file_reproj = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/weather_reproj.tif')

with rasterio.open(meteo_file_reproj) as src:
    bounds = src.bounds  # (left, bottom, right, top)
    
# Create the grid coordinates
x_steps_left = np.ceil(np.abs(landsat_tile_195_27[0] - bounds[0])/grid_size)
x_steps_right = np.ceil(np.abs(landsat_tile_195_27[0] - bounds[2])/grid_size)
y_steps_bottom = np.ceil(np.abs(landsat_tile_195_27[1] - bounds[1])/grid_size)
y_steps_top = np.ceil(np.abs(landsat_tile_195_27[1] - bounds[3])/grid_size)



x_vals = np.arange(landsat_tile_195_27[0]-grid_size*x_steps_left, landsat_tile_195_27[0]+grid_size*(x_steps_right+1), grid_size)
y_vals = np.arange(landsat_tile_195_27[1]+grid_size*y_steps_top, landsat_tile_195_27[1]-grid_size*(y_steps_bottom+1), -grid_size)  # Decreasing y for top-down order

# Crop grid to cover jsut the extent of the meteo tif 
x_start_idx = np.where(x_vals < bounds[0])[0][-1] 
x_end_idx = np.where(x_vals > bounds[2])[0][0]  
y_start_idx = np.where(y_vals > bounds[3])[0][-1]  
y_end_idx = np.where(y_vals < bounds[1])[0][0] 

x_vals = x_vals[x_start_idx:x_end_idx+1]
y_vals = y_vals[y_start_idx:y_end_idx+1]


# Create a meshgrid of coordinates
X, Y = np.meshgrid(x_vals, y_vals)

# Flatten the grid for easier processing
grid_points = np.column_stack([X.ravel(), Y.ravel()])

# Convert to geopandas and grid shapefile
polygons = []
attributes = []

for i in range(len(x_vals) - 1):
    for j in range(len(y_vals) - 1):
        left, right = x_vals[i], x_vals[i + 1]
        bottom, top = y_vals[j + 1], y_vals[j]

        # Create a square polygon
        polygon = Polygon([(left, bottom), (right, bottom), (right, top), (left, top), (left, bottom)])
        polygons.append(polygon)

        # Store attributes
        attributes.append({"top": top, "left": left, "right": right, "bottom": bottom})

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(attributes, geometry=polygons, crs=src.crs)

# Save to Shapefile
output_shapefile = "grid_landsat.shp"
gdf.to_file(output_shapefile)
