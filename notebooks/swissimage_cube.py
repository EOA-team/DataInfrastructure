import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import rioxarray
from pyproj import Transformer
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, box
from rasterio.enums import Resampling



def coords_to_topleft(ds, res):
  # Adjust coords to topleft corner of pixel
  x_coords = ds['x']
  y_coords = ds['y']

  ds = ds.assign_coords({'x': x_coords - res / 2, 'y': y_coords + res / 2})
  return ds

def round_to_nearest_even(number):
    rounded = round(number)
    if rounded % 2 == 0:
        return rounded
    else:
        return rounded + 1 if number > rounded else rounded - 1

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

def get_bounds_topleft_ds(ds, res):
  """
  Get bounds given that the coords are the topleft corner of the pixel
  """
  return ds.x.min().item(), ds.y.min().item() - res, ds.x.max().item() + res, ds.y.max().item()

def extract_bounds_year(file):
    parts = file.split('_')
    minx = int(parts[1])
    maxx = minx + 1280
    maxy = int(parts[2])
    miny = maxy - 1280
    yr = int(parts[3][:4])
    return minx, miny, maxx, maxy, yr

# Function to check intersection
def intersects_with_bbox(row, bbox_aoi):
    file_polygon = box(row['minx'], row['miny'], row['maxx'], row['maxy'])
    return bbox_aoi.intersects(file_polygon)



# SwissImage

file_path = os.path.expanduser('~/mnt/eo-nas1/data/swisstopo/SwissImage_2m/swissimage-dop10_2022_2681-1253_2_2056.tif')
file_path2 = os.path.expanduser('~/mnt/eo-nas1/data/swisstopo/SwissImage_2m/swissimage-dop10_2022_2682-1253_2_2056.tif')
file_path3 = os.path.expanduser('~/mnt/eo-nas1/data/swisstopo/SwissImage_2m/swissimage-dop10_2022_2681-1252_2_2056.tif')
file_path4 = os.path.expanduser('~/mnt/eo-nas1/data/swisstopo/SwissImage_2m/swissimage-dop10_2022_2682-1252_2_2056.tif')
swissimage = rioxarray.open_rasterio(file_path)
swissimage2 = rioxarray.open_rasterio(file_path2)
swissimage3 = rioxarray.open_rasterio(file_path3)
swissimage4 = rioxarray.open_rasterio(file_path4)


# Shift coords from center to topleft of pixel
swissimage = coords_to_topleft(swissimage, 2)
swissimage2 = coords_to_topleft(swissimage2, 2)
swissimage3 = coords_to_topleft(swissimage3, 2)
swissimage4 = coords_to_topleft(swissimage4, 2)

# Reproject to EPSG 32632 (nearest by default)
swissimage = swissimage.rio.reproject("EPSG:32632", shape=(500,500), resampling=Resampling.cubic) # TODO: doesnt conserve the shape of the data
swissimage2 = swissimage2.rio.reproject("EPSG:32632", shape=(500,500), resampling=Resampling.cubic)
swissimage3 = swissimage3.rio.reproject("EPSG:32632", shape=(500,500), resampling=Resampling.cubic)
swissimage4 = swissimage4.rio.reproject("EPSG:32632", shape=(500,500), resampling=Resampling.cubic)


# Resample to custom grid aligned with S2 data

start_x = round_to_lower_even(swissimage['x'].values[0]) 
end_x = round_to_higher_even(swissimage['x'].values[-1])
start_y = round_to_higher_even(swissimage['y'].values[0])
end_y = round_to_lower_even(swissimage['y'].values[-1])

new_x = np.arange(start_x, end_x+2, 2) # TO DO: then lengths should be fixed --> Actually should align to custom grid here!
new_y = np.arange(start_y, end_y-2, -2)

# Interpolate data
swissimage = swissimage.interp(x = new_x, y = new_y, method = "nearest")
swissimage.rio.to_raster('si1_reproj_pipeline.tif')
"""
swissimage_nearest = swissimage.interp(x = new_x, y = new_y, method = "nearest")
swissimage_nearest.rio.to_raster('swissimage_reproj_nearest.tif')
swissimage_cubic = swissimage.interp(x = new_x, y = new_y, method = "cubic", kwargs={'fill_value':'extrapolate'})
swissimage_nearest.rio.to_raster('swissimage_reproj_cubic.tif')
"""


start_x = round_to_lower_even(swissimage2['x'].values[0]) # TODO adapt use of higher or lower even nbr better
end_x = round_to_higher_even(swissimage2['x'].values[-1])
start_y = round_to_higher_even(swissimage2['y'].values[0])
end_y = round_to_lower_even(swissimage2['y'].values[-1])

new_x = np.arange(start_x, end_x+2, 2) # TO DO: then lengths should be fixed --> Actually should align to custom grid here!
new_y = np.arange(start_y, end_y-2, -2)

# Interpolate data
swissimage2 = swissimage2.interp(x = new_x, y = new_y, method = "nearest")
swissimage2.rio.to_raster('si2_reproj_pipeline.tif')
"""
swissimage2_nearest = swissimage2.interp(x = new_x, y = new_y, method = "nearest")
swissimage2_nearest.rio.to_raster('swissimage2_reproj_nearest.tif')
swissimage2_cubic = swissimage2.interp(x = new_x, y = new_y, method = "cubic", kwargs={'fill_value':'extrapolate'})
swissimage2_nearest.rio.to_raster('swissimage2_reproj_cubic.tif')
"""

start_x = round_to_lower_even(swissimage3['x'].values[0]) # TODO adapt use of higher or lower even nbr better
end_x = round_to_higher_even(swissimage3['x'].values[-1])
start_y = round_to_higher_even(swissimage3['y'].values[0])
end_y = round_to_lower_even(swissimage3['y'].values[-1])
new_x = np.arange(start_x, end_x+2, 2) # TO DO: then lengths should be fixed --> Actually should align to custom grid here!
new_y = np.arange(start_y, end_y-2, -2)
swissimage3 = swissimage3.interp(x = new_x, y = new_y, method = "nearest")
swissimage3.rio.to_raster('si3_reproj_pipeline.tif')

start_x = round_to_lower_even(swissimage4['x'].values[0]) # TODO adapt use of higher or lower even nbr better
end_x = round_to_higher_even(swissimage4['x'].values[-1])
start_y = round_to_higher_even(swissimage4['y'].values[0])
end_y = round_to_lower_even(swissimage4['y'].values[-1])
new_x = np.arange(start_x, end_x+2, 2) # TO DO: then lengths should be fixed --> Actually should align to custom grid here!
new_y = np.arange(start_y, end_y-2, -2)
swissimage4 = swissimage4.interp(x = new_x, y = new_y, method = "nearest")
swissimage4.rio.to_raster('si4_reproj_pipeline.tif')


# Find cube to register (corresponding AOI)

data_folder = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH')
cubes = [f for f in os.listdir(data_folder) if f.endswith('zarr')]
df_cubes = pd.DataFrame(cubes, columns=['file'])


# Apply the function to extract minx and maxy into new columns
df_cubes[['minx', 'miny', 'maxx', 'maxy', 'yr']] = df_cubes['file'].apply(lambda x: pd.Series(extract_bounds_year(x)))


# Filter files by bbox
minx, miny, maxx, maxy = get_bounds_topleft_ds(swissimage, 2)
bbox_aoi = box(minx, miny, maxx, maxy)

filtered_files = df_cubes[df_cubes.apply(intersects_with_bbox, axis=1, bbox_aoi=bbox_aoi)]

# Filter files by year
filtered_files = filtered_files[filtered_files['yr'] == 2023]
filtered_filenames = filtered_files['file'].tolist()

"""
minx, miny, maxx, maxy = get_bounds_topleft_ds(swissimage, 2)
si = Polygon([(maxx, maxy), (maxx, miny), (minx, miny), (minx, maxy), (maxx, maxy)])

minx, miny, maxx, maxy = get_bounds_topleft_ds(swissimage2, 2)
si2 = Polygon([(maxx, maxy), (maxx, miny), (minx, miny), (minx, maxy), (maxx, maxy)])

minx, miny, maxx, maxy = get_bounds_topleft_ds(swissimage3, 2)
si3 = Polygon([(maxx, maxy), (maxx, miny), (minx, miny), (minx, maxy), (maxx, maxy)])

minx, miny, maxx, maxy = get_bounds_topleft_ds(swissimage4, 2)
si4 = Polygon([(maxx, maxy), (maxx, miny), (minx, miny), (minx, maxy), (maxx, maxy)])

gdf_si = gpd.GeoDataFrame(geometry=[si], crs=32632)
gdf_si2 = gpd.GeoDataFrame(geometry=[si2], crs=32632)
gdf_si3 = gpd.GeoDataFrame(geometry=[si3], crs=32632)
gdf_si4 = gpd.GeoDataFrame(geometry=[si4], crs=32632)


minx, miny, maxx, maxy = filtered_files.reset_index(drop=True).iloc[0][["minx", "miny", "maxx", "maxy"]]
cube1 = Polygon([(maxx, maxy), (maxx, miny), (minx, miny), (minx, maxy), (maxx, maxy)])

minx, miny, maxx, maxy = filtered_files.reset_index(drop=True).iloc[1][["minx", "miny", "maxx", "maxy"]]
cube2 = Polygon([(maxx, maxy), (maxx, miny), (minx, miny), (minx, maxy), (maxx, maxy)])

minx, miny, maxx, maxy = filtered_files.reset_index(drop=True).iloc[2][["minx", "miny", "maxx", "maxy"]]
cube3 = Polygon([(maxx, maxy), (maxx, miny), (minx, miny), (minx, maxy), (maxx, maxy)])

minx, miny, maxx, maxy = filtered_files.reset_index(drop=True).iloc[3][["minx", "miny", "maxx", "maxy"]]
cube4 = Polygon([(maxx, maxy), (maxx, miny), (minx, miny), (minx, maxy), (maxx, maxy)])
"""

# Get SwissImage data to fill in the cube
swissimage = swissimage.to_dataset(name='bands')
swissimage2 = swissimage2.to_dataset(name='bands')
swissimage3 = swissimage3.to_dataset(name='bands')
swissimage4 = swissimage4.to_dataset(name='bands')

print(swissimage.coords)
print(swissimage2.coords)
print(swissimage3.coords)
print(swissimage4.coords)

#all_si = swissimage.combine_first(swissimage2).combine_first(swissimage3).combine_first(swissimage4)
#all_si = xr.merge([swissimage, swissimage2, swissimage3, swissimage4], compat="override")
datasets = [swissimage, swissimage2, swissimage3, swissimage4]
stacked = xr.concat(datasets, dim='stacked')
mean = stacked.mean(dim='stacked', skipna=True)
all_si = xr.where(stacked.notnull().any(dim='stacked'), stacked.max(dim='stacked'), mean) # Use non-NaN values if available; otherwise, use the mean
    
plt.imshow(all_si.isel(band=0).bands.values)
plt.savefig('sicube.png')
"""
gdf_cube1 = gpd.GeoDataFrame(geometry=[cube1], crs=32632)
gdf_cube2 = gpd.GeoDataFrame(geometry=[cube2], crs=32632)
gdf_cube3 = gpd.GeoDataFrame(geometry=[cube3], crs=32632)
gdf_cube4 = gpd.GeoDataFrame(geometry=[cube4], crs=32632)

f, ax = plt.subplots(1,1,figsize=(6,6))
gdf_si.plot(ax=ax, color='r', alpha=0.5)
gdf_si2.plot(ax=ax, color='r', alpha=0.55)
gdf_si3.plot(ax=ax, color='r', alpha=0.6)
gdf_si4.plot(ax=ax, color='r', alpha=0.65)
gdf_cube1.plot(ax=ax, color='yellow', alpha=0.7)
gdf_cube2.plot(ax=ax, color='yellow', alpha=0.65)
gdf_cube3.plot(ax=ax, color='yellow', alpha=0.6)
gdf_cube4.plot(ax=ax, color='b', alpha=0.55)

plt.savefig('overlap.png')
"""