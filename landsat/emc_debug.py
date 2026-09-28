import os
from pathlib import Path
import sys

base_dir = Path(os.path.dirname(os.path.realpath("__file__"))).parent.parent
sys.path.insert(0, str(base_dir))
import earthnet_minicuber as emc

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, box
from shapely.ops import cascaded_union
import matplotlib.pyplot as plt
import contextily as cx
import numpy as np
import zarr
import pickle
import time
import datetime
import concurrent.futures
import threading
import gc
import rioxarray
import time



specs = {
        "lon_lat": (511830, 5267580), # topleft
        "xy_shape": (128*4, 128*4), # width, height of cutout around center pixel
        "resolution": 30, # in meters.. will use this on a local UTM grid..
        "time_interval": "2022-03-01/2022-03-31",
        "final_epsg": 32632,
        "providers": [
            {
                "name": "ls",
                "kwargs": {
                    "bands": ["coastal", "blue", "green", "red", "nir08", "swir16", "lwir", "swir22", "lwir11", "urad", "drad", "trad",  "emis", "emsd", "atran", "cdist", "qa_pixel", "qa_aerosol", "qa_radsat", "qa", "cloud_qa", "atmos_opacity"],
                    "data_source": "planetary_computer",
                    }
            }
            ]
    }

start = time.time()
cube = emc.load_minicube(specs, compute=True, verbose=True)
end = time.time()
print('Took', end-start)

"""
#cube.to_zarr('test.zarr', consolidated=True, mode='w')

import zarr
data_path = os.path.expanduser('~/mnt/eo-nas1/data/satellite/landsat/raw/CH/89/LS_262230_5125500_20190104_20191222.zarr')
red = np.squeeze(zarr.open(data_path + '/OLI_B4', mode='r'))
print(red)

#grid_landsat = '~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/grid_landsat_CH.shp'
#grid = gpd.read_file(os.path.expanduser(grid_landsat))
#print(len(grid))
"""
