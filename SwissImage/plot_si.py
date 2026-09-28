import xarray as xr
import os
import matplotlib.pyplot as plt
import numpy as np
import rioxarray

### LOAD DATA
data_folder = os.path.expanduser('~/mnt/eo-nas1/data/swisstopo/SwissImage/cubes/10cm/')
#files = [os.path.join(data_folder, f) for f in os.listdir(data_folder)]


files = ['/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/swisstopo/SwissImage/cubes/10cm/SwissImage0.1_297180_5179820.zarr',\
'/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/swisstopo/SwissImage/cubes/10cm/SwissImage0.1_295900_5149100.zarr',\
'/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/swisstopo/SwissImage/cubes/10cm/SwissImage0.1_295900_5150380.zarr']



""" 
['/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/swisstopo/SwissImage/cubes/10cm/SwissImage0.1_265180_5117100.zarr',\
  '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/swisstopo/SwissImage/cubes/10cm/SwissImage0.1_265180_5115820.zarr']

['/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/swisstopo/SwissImage/cubes/10cm/SwissImage0.1_272860_5115820.zarr',\
'/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/swisstopo/SwissImage/cubes/10cm/SwissImage0.1_274140_5150380.zarr',\
'/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/swisstopo/SwissImage/cubes/10cm/SwissImage0.1_274140_5143980.zarr',\
'/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/swisstopo/SwissImage/cubes/10cm/SwissImage0.1_274140_5147820.zarr',\
'/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/swisstopo/SwissImage/cubes/10cm/SwissImage0.1_274140_5149100.zarr']

['/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/swisstopo/SwissImage/cubes/10cm/SwissImage0.1_270300_5119660.zarr', \
'/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/swisstopo/SwissImage/cubes/10cm/SwissImage0.1_270300_5118380.zarr', \
'/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/swisstopo/SwissImage/cubes/10cm/SwissImage0.1_270300_5115820.zarr', \
'/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/swisstopo/SwissImage/cubes/10cm/SwissImage0.1_270300_5114540.zarr']


"""


#mc_timeseries = xr.open_mfdataset(files).compute()
for i, f in enumerate(files):
  mc_timeseries = xr.open_zarr(f).compute()
  print(mc_timeseries.x.values[0], mc_timeseries.x.values[1],mc_timeseries.x.values[-1],mc_timeseries.y.values[0], mc_timeseries.y.values[1],mc_timeseries.y.values[-1])
  print(mc_timeseries.x.values[0]-mc_timeseries.x.values[1],mc_timeseries.y.values[0]-mc_timeseries.y.values[1])
  #mc_timeseries[['R', 'G', 'B']].rio.to_raster(f'plot_si{i}.tif')

"""
### PLOT DATA
r = mc_timeseries['R'].values
g = mc_timeseries['G'].values
b = mc_timeseries['B'].values
rgb_image = np.stack([r, g, b], axis=-1)
plt.imshow(rgb_image)
plt.savefig('si_view.png')
"""
