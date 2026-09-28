import os
import xarray as xr
import rioxarray
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import shutil
import time


data_dir = os.path.expanduser('~/mnt/eo-nas1/data/meteo/snowdepth')
all_files = [f for f in os.listdir(data_dir) if f.endswith('zarr')]

for i, f in enumerate(all_files): 
    print(f'Processing {i+1}/{len(all_files)}: {f}')
    start = time.time()
    zarr_path = os.path.join(data_dir, f)
    try:
        ds = xr.open_zarr(zarr_path)
    except:
        print('Cant open file')
        continue

    # --- Drop duplicate timestamps efficiently ---
    # get unique time indices without loading the whole dataset
    _, unique_idx = np.unique(ds['time'], return_index=True)
    if len(unique_idx) != ds.sizes['time']:
        print('Removing duplicates...')
        ds = ds.isel(time=np.sort(unique_idx)).chunk({"time": -1})
      
        # --- Overwrite the Zarr store efficiently ---
        tmp_path = zarr_path + '.tmp'
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
        ds.to_zarr(tmp_path, mode='w', consolidated=True) #, safe_chunk=False)
        shutil.rmtree(zarr_path)
        os.rename(tmp_path, zarr_path)

        print(f"Removed duplicates and rewrote: {f}, ({time.time()-start})")
    else:
        print(f"No duplicates found in {f}")
    