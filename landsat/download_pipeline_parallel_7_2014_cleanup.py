import os
from pathlib import Path
import sys

base_dir = Path(os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure'))
sys.path.insert(0, str(base_dir))
import earthnet_minicuber as emc

import geopandas as gpd
import pandas as pd
import numpy as np
import zarr
import time
import datetime
import concurrent.futures
import gc


TARGET_YEARS = {2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022}
OUTPUT_PREFIX = os.path.expanduser('~/mnt/eo-nas1/data/satellite/landsat/raw/CH/7/')
GRID_PATH     = '~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/grid_landsat_CH.shp'
MAX_WORKERS   = 30   # flat (tile, year) pool — tune to PC rate limits
RESOLUTION    = 30
PATCH_PX      = 128
PATCH_M       = PATCH_PX * RESOLUTION  # 3840 m


def extract_minx_maxy(file):
    parts = file.split('_')
    return int(parts[1]), int(parts[2]), int(parts[3][:4])


def extract_date(t):
    year  = t.astype('datetime64[Y]').astype(int) + 1970
    month = (t.astype('datetime64[M]').astype(int) % 12) + 1
    day   = (t.astype('datetime64[D]').astype(int) -
             t.astype('datetime64[M]').astype('datetime64[D]').astype(int)) + 1
    return year, f"{month:02d}", f"{day:02d}"


def save_patch(cube, output_prefix):
    lat_max = cube.lat.max().values
    lon_min = cube.lon.min().values
    time_min, time_max = cube.time.min().values, cube.time.max().values
    ys, ms, ds = extract_date(time_min)
    ye, me, de = extract_date(time_max)
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    output_path = output_prefix + f'LS_{int(lon_min)}_{int(lat_max)}_{ys}{ms}{ds}_{ye}{me}{de}.zarr'
    if not os.path.exists(output_path):
        cube.to_zarr(output_path, consolidated=True, mode='w',
                     encoding={v: {'compressor': compressor} for v in cube.data_vars})
        print(f'  Saved {output_path}')
    return output_path


def download_tile_year(left, top, year, specs_base):
    """Download one 128×128 tile for one year. Returns (left, top, year, success)."""
    try:
        specs = dict(specs_base)  # thread-local copy
        specs['lon_lat']       = (left, top)
        specs['xy_shape']      = (PATCH_PX, PATCH_PX)
        specs['time_interval'] = f'{year}-01-01/{year}-12-31'

        print(f"{datetime.datetime.now()}: ({left}, {top}) year={year}")
        t0 = time.time()
        cube = emc.load_minicube(specs, compute=True, verbose=False)
        print(f"  ({left}, {top}) {year} downloaded in {time.time()-t0:.0f}s")

        if cube is not None:
            save_patch(cube, OUTPUT_PREFIX)
        return left, top, year, True

    except Exception as e:
        print(f"  ERROR ({left}, {top}) {year}: {e}")
        return left, top, year, False


def build_missing_jobs(grid):
    """Return list of (left, top, year) that are missing on disk."""
    on_disk = {}
    for f in os.listdir(OUTPUT_PREFIX):
        if not f.endswith('.zarr'):
            continue
        try:
            lx, ty, yr = extract_minx_maxy(f)
            on_disk.setdefault((lx, ty), set()).add(yr)
        except Exception:
            continue

    jobs = []
    for _, row in grid.iterrows():
        left, top = int(row['left']), int(row['top'])
        years_on_disk = on_disk.get((left, top), set())
        missing = TARGET_YEARS - years_on_disk
        for yr in sorted(missing):
            jobs.append((left, top, yr))

    print(f"Found {len(jobs)} missing (tile, year) jobs across "
          f"{len(set((l,t) for l,t,_ in jobs))} tiles")
    return jobs


if __name__ == '__main__':
    grid = gpd.read_file(GRID_PATH)

    specs_base = {
        'lon_lat':      (None, None),
        'xy_shape':     (None, None),
        'resolution':   RESOLUTION,
        'time_interval': '2014-01-01/2014-12-31',
        'final_epsg':   32632,
        'providers': [{
            'name': 'ls',
            'kwargs': {
                'bands': ['blue', 'green', 'red', 'nir08', 'swir16', 'lwir', 'swir22',
                          'urad', 'drad', 'trad', 'emis', 'emsd', 'atran', 'cdist',
                          'qa_pixel', 'qa_radsat', 'qa', 'cloud_qa', 'atmos_opacity'],
                'data_source': 'planetary_computer',
                'platforms':   ['landsat-7'],
            }
        }]
    }

    jobs = build_missing_jobs(grid)
    if not jobs:
        print('Nothing missing — all done.')
        raise SystemExit(0)

    failed = []
    # Retry loop — keep going until all succeed
    while jobs:
        print(f"\n{datetime.datetime.now()} — submitting {len(jobs)} jobs "
              f"with {MAX_WORKERS} workers")
        next_failed = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(download_tile_year, l, t, y, specs_base): (l, t, y)
                       for l, t, y in jobs}
            for fut in concurrent.futures.as_completed(futures):
                l, t, y, ok = fut.result()
                if not ok:
                    next_failed.append((l, t, y))
        jobs = next_failed
        if jobs:
            print(f"  {len(jobs)} jobs failed, retrying …")

    print('\nAll missing tiles downloaded.')
