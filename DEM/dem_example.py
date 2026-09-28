#!/.virtualenvs/002_GrassSense/bin/python
# -*- coding: utf-8 -*-

#%%
import numpy as np
import matplotlib.pyplot as plt
font = {'family' : 'DejaVu Sans',
        'size'   : 12}
from matplotlib import rc
rc('font', **font)
import zarr
import glob
import xdem # install in your venv with 'pip install xdem'

# tiles path
zarr_root_path = '/mnt/eo-nas1/data/swisstopo/DEM/'

def show_im_(im,fname):
    plt.figure()
    plt.imshow(im)
    plt.colorbar()
    plt.show()
    plt.savefig(fname)

# import tile
f_list = glob.glob(zarr_root_path + 'sa3d*')
zarr_path = '/home/f80873755@agsad.admin.ch/mnt/eo-nas1/data/swisstopo/DEM_new/sa3d_277980_5147820.zarr' # f_list[240] # FIRST ZARR FILE, YOU CAN TRY OTHERS THE ARTIFACT IS PRESENT IN SOME OF THEM

# missing data value
z =  zarr.open(zarr_path,mode='r')
nodata = z.attrs['nodata']

# dem array
dem = zarr.open(zarr_path + '/height',mode='r')[:] # get tile coordinate vectors
dem[dem==nodata] = np.nan

# save dem
show_im_(dem,'dem.png')

# example of derivative to show the artifact: dem diff
dem_diff = np.diff(dem)

# save dem diff
show_im_(dem_diff,'dem_diff.png')



