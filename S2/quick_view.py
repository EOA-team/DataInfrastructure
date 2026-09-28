""" 
Quickly view the data in a zarr store
Will show RGB image of cube for the least cloudy date

Sélène Ledain
Aug 15th 2024
"""

from argparse import ArgumentParser
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np



def view(cube_path):
  """ 
  Plot RGB image of cube on least cloudy date

  :param cube_path: str, patht to zarr sotre
  """
  dataset = xr.open_zarr(cube_path).compute()
  cloud_coverage = (dataset['s2_mask'] != 0).mean(dim=['lat', 'lon'])
  least_cloudy_date = cloud_coverage.argmin()
  rgb = dataset.isel(time=least_cloudy_date)[['s2_B04', 's2_B03', 's2_B02']].to_dataarray('band')

  # Normalize the data for better visualization (optional)
  rgb = rgb.where(rgb != 65535, np.nan)
  rgb = rgb.where(~np.isnan(rgb), other=1.0)
  rgb_normalized = rgb / rgb.max()
  # Plot the RGB image
  plt.figure(figsize=(10, 10))
  plt.imshow(rgb_normalized.transpose('lat', 'lon', 'band'))
  plt.title(f"Least Cloudy date for {cube_path.split('/')[-1].split('.zarr')[0]}")
  plt.axis('off')
  plt.savefig(f"quick_view_{cube_path.split('/')[-1].split('.zarr')[0]}.png")
  plt.show()

  return 


if __name__ ==  "__main__":

  parser = ArgumentParser()
  parser.add_argument('cube_path', type = str, metavar='path/to/cube.zarr', help='path to the zarr store')
  args = parser.parse_args()

  view(args.cube_path)