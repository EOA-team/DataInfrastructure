""" 
Coregistration pipeline using SwissImage to correct S2 data
"""

import os
import xarray as xr
import numpy as np
import geopandas as gpd
import glob
import warnings
warnings.filterwarnings('ignore')
from pyproj import CRS
from shapely import Polygon
import zarr
import datetime
import pandas as pd
from collections import defaultdict
import rioxarray
from PIL import Image
from scipy.ndimage import affine_transform
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator


def analyse_stack(target_folder, reference_folder, output_folder, minx, maxy):

    processed_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.zarr')]
    original_files = [os.path.join(target_folder, f) for f in os.listdir(output_folder) if f.endswith('.zarr')]

    # Find a file that was processed, and its original counterpart
    f_coreg = [f for f in processed_files if f'S2_{minx}_{maxy}' in f][0].split('/')[-1]
    f_orig = [f for f in os.listdir(output_folder) if f.endswith('.zarr') and f_coreg.split('.zarr')[0] in f][0]

    ds_orig = xr.open_zarr(os.path.join(target_folder, f_orig)).compute()
    ds_coreg = xr.open_zarr(os.path.join(output_folder, f_coreg)).compute()
    si = xr.open_zarr(os.path.join(reference_folder, f'SwissImage0.1_{f_orig.split("_")[1]}_{f_orig.split("_")[2]}.zarr')).compute()

    plot_mp4(si, ds_orig, ds_coreg, 'coreg.mp4')


def plot_imgs(ref, ds, ds_coreg, i, outpath):
  scale_factor = 1.0 / 10000.0  # Scale factor for DN to [0, 1]
  r = ds_coreg['s2_B04'] * scale_factor
  g = ds_coreg['s2_B03'] * scale_factor
  b = ds_coreg['s2_B02'] * scale_factor

  # Stack bands into an RGB array
  rgb_coreg = xr.concat([r, g, b], dim='band').transpose('time', 'lat', 'lon', 'band')
  rgb_coreg = rgb_coreg.where(~np.isnan(rgb_coreg), other=1.0)

  r = ds['s2_B04'] * scale_factor
  g = ds['s2_B03'] * scale_factor
  b = ds['s2_B02'] * scale_factor

  # Stack bands into an RGB array
  rgb = xr.concat([r, g, b], dim='band').transpose('time', 'lat', 'lon', 'band')
  rgb = rgb.where(~np.isnan(rgb), other=1.0)

  ref_rgb = ref[['R','G','B']].to_array().values.transpose(1,2,0)
    
  f, axs = plt.subplots(1, 3, figsize=(15, 5))

  axs[0].imshow(ref_rgb)
  axs[0].set_xticks(np.arange(ref_rgb.shape[1]))
  axs[0].set_yticks(np.arange(ref_rgb.shape[0]))
  axs[0].set_xticklabels(ref.coords['x'].values)
  axs[0].set_yticklabels(ref.coords['y'].values)
  axs[0].xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=6))
  axs[0].yaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=6))
  plt.setp(axs[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

  axs[1].imshow(rgb[i].values*3)
  axs[1].set_xticks(np.arange(rgb.sizes['lon']))
  axs[1].set_yticks(np.arange(rgb.sizes['lat']))
  axs[1].set_xticklabels(rgb.coords['lon'].values)
  axs[1].set_yticklabels(rgb.coords['lat'].values)
  axs[1].xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=6))
  axs[1].yaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=6))
  plt.setp(axs[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

  axs[2].imshow(rgb_coreg[i].values*3)
  axs[2].set_xticks(np.arange(rgb_coreg.sizes['lon']))
  axs[2].set_yticks(np.arange(rgb_coreg.sizes['lat']))
  axs[2].set_xticklabels(rgb_coreg.coords['lon'].values)
  axs[2].set_yticklabels(rgb_coreg.coords['lat'].values)
  axs[2].xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=6))
  axs[2].yaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=6))
  plt.setp(axs[2].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
  plt.savefig(outpath)

  return


def plot_gif(ds, ds_coreg, outpath):
  
  """ 
  rgb_coreg = ds_coreg[['s2_B04','s2_B03','s2_B02']].astype(float)

  # Need to rescale each band to 0-255 and then set the nan values to 255
  rgb_coreg = rgb_coreg.where(rgb_coreg != 65535, np.nan)
  max_vals = rgb_coreg.max(dim=['time', 'lat', 'lon'], skipna=True)
  min_vals = rgb_coreg.min(dim=['time', 'lat', 'lon'], skipna=True)
  rgb_scaled = ((rgb_coreg - min_vals) / (max_vals - min_vals)) * 255.0
  rgb_coreg = rgb_scaled.where(~rgb_scaled.isnull(), 0)
  """

  scale_factor = 1.0 / 10000.0  # Scale factor for DN to [0, 1]
  r = ds_coreg['s2_B04'] * scale_factor
  g = ds_coreg['s2_B03'] * scale_factor
  b = ds_coreg['s2_B02'] * scale_factor

  # Stack bands into an RGB array
  rgb_coreg = xr.concat([r, g, b], dim='band').transpose('time', 'lat', 'lon', 'band')
  rgb_coreg = rgb_coreg.where(~np.isnan(rgb_coreg), other=1.0)

  """ 
  rgb = ds[['s2_B04','s2_B03','s2_B02']].astype(float)

  # Need to rescale each band to 0-255 and then set the nan values to 255
  rgb = rgb.where(rgb != 65535, np.nan)
  max_vals = 10000 #rgb.max(dim=['time', 'lat', 'lon'], skipna=True)
  min_vals = 0 #rgb.min(dim=['time', 'lat', 'lon'], skipna=True)
  rgb_scaled = ((rgb - min_vals) / (max_vals - min_vals)) * 255.0
  rgb = rgb_scaled.where(~rgb_scaled.isnull(), 0)
  """

  r = ds['s2_B04'] * scale_factor
  g = ds['s2_B03'] * scale_factor
  b = ds['s2_B02'] * scale_factor

  # Stack bands into an RGB array
  rgb = xr.concat([r, g, b], dim='band').transpose('time', 'lat', 'lon', 'band')
  rgb = rgb.where(~np.isnan(rgb), other=1.0)
  print(rgb)


  # Create a PIL Image object from the numpy array
  gif = []
  for t in range(rgb_coreg.sizes['time']):
      img_orig = rgb[t] #.isel(time=t).to_array().values.transpose(1,2,0)
      im_rescaled_orig = img_orig*5

      img_coreg = rgb_coreg[t] #.isel(time=t).to_array().values.transpose(1,2,0)
      im_rescaled_coreg = img_coreg*5

      # Combine the original and coregistered images side by side
      combined_img = np.hstack((im_rescaled_orig, im_rescaled_coreg))

      # Convert numpy array to PIL Image and resize
      pil_img = Image.fromarray(combined_img.astype("uint8")).resize((1600, 600))  # Resize to twice the width
      gif.append(pil_img)


  # Save the GIF
  gif[0].save(outpath,
              save_all=True,
              append_images=gif[1:],
              duration=500,  # Set duration between frames in milliseconds
              loop=1)  # Set loop to 0 for infinite looping
  return


def update(frame):
    # Update images for the current timestamp
    im_rgb.set_array(rgb[frame].values*3)
    im_rgbcoreg.set_array(rgb_coreg[frame].values*3)
    # Set the titles on the axes
    axs[1].set_title(f'Orig {str(rgb.time.values[frame]).split("T")[0]}')
    axs[2].set_title(f'Coreg {str(rgb_coreg.time.values[frame]).split("T")[0]}')
    
    return im_rgb, im_rgbcoreg


def plot_mp4(ref, ds, ds_coreg, outpath):
  global rgb, rgb_coreg, im_rgb, im_rgbcoreg, axs
  scale_factor = 1.0 / 10000.0  # Scale factor for DN to [0, 1]
  r = ds_coreg['s2_B04'] * scale_factor
  g = ds_coreg['s2_B03'] * scale_factor
  b = ds_coreg['s2_B02'] * scale_factor

  # Stack bands into an RGB array
  rgb_coreg = xr.concat([r, g, b], dim='band').transpose('time', 'lat', 'lon', 'band')
  rgb_coreg = rgb_coreg.where(~np.isnan(rgb_coreg), other=1.0)

  r = ds['s2_B04'] * scale_factor
  g = ds['s2_B03'] * scale_factor
  b = ds['s2_B02'] * scale_factor

  # Stack bands into an RGB array
  rgb = xr.concat([r, g, b], dim='band').transpose('time', 'lat', 'lon', 'band')
  rgb = rgb.where(~np.isnan(rgb), other=1.0)

  ref_rgb = ref[['R','G','B']].to_array().values.transpose(1,2,0)

  # Set up the figure and axis
  fig, axs = plt.subplots(1, 3, figsize=(15, 5))

  # Initialize images
  img_ref = axs[0].imshow(ref_rgb)
  im_rgb = axs[1].imshow(rgb[0].values*3)
  im_rgbcoreg = axs[2].imshow(rgb_coreg[0].values*3)

  # Set titles and axis labels
  axs[0].set_title('Ref')
  axs[1].set_title(f'Orig {str(rgb.time.values[0]).split("T")[0]}')
  axs[2].set_title(f'Coreg {str(rgb_coreg.time.values[0]).split("T")[0]}')

  # Create the animation
  ani = FuncAnimation(fig, update, frames=len(rgb.time), blit=True)

  # Save the animation as an MP4 video using ffmpeg
  ani.save(outpath, writer='ffmpeg', fps=2)  # Adjust fps (frames per second) as needed

  return


def plot_single_gif(ds, outpath):
  
  rgb = ds[['s2_B04','s2_B03','s2_B02']].astype(float)

  # Need to rescale each band to 0-255 and then set the nan values to 255
  rgb = rgb.where(rgb != 65535, np.nan)
  max_vals = rgb.max(dim=['time', 'lat', 'lon'])
  min_vals = rgb.min(dim=['time', 'lat', 'lon'])
  rgb_scaled = ((rgb - min_vals) / (max_vals - min_vals)) * 255.0
  rgb = rgb_scaled.where(~rgb_scaled.isnull(), 0)

  # Create a PIL Image object from the numpy array
  gif = []
  for t in range(rgb.sizes['time']):

      # CHeck clouds < 50%
      scl = ds.isel(time=t).s2_SCL
      scl_mask = xr.where(scl.isin([0,1,2,3,7,8,9,10]), True, False) 
      cloud = ds.isel(time=t).s2_mask
      cloud_mask = xr.where(cloud != 0, True, False)
      data_mask = scl_mask & cloud_mask

      if not data_mask.values.sum() > (data_mask.shape[0]*data_mask.shape[1])//2: 
  
        img_orig = rgb.isel(time=t).to_array().values.transpose(1,2,0)
        im_rescaled_orig = img_orig*5

        # Convert numpy array to PIL Image and resize
        pil_img = Image.fromarray(im_rescaled_orig.astype("uint8")).resize((600, 600))  # Resize to twice the width
        gif.append(pil_img)


  # Save the GIF
  gif[0].save(outpath,
              save_all=True,
              append_images=gif[1:],
              duration=500,  # Set duration between frames in milliseconds
              loop=0)  # Set loop to 0 for infinite looping
  return


if __name__ == "__main__":

  target_folder = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH')
  reference_folder = os.path.expanduser('~/mnt/eo-nas1/data/swisstopo/SwissImage/cubes/10cm')
  output_folder = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/coreg/CH')
  minx = 444380
  maxy = 5236140

  analyse_stack(target_folder, reference_folder, output_folder, minx, maxy) # will create mp4 file called coreg.mp4 in working directory