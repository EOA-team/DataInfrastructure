import os
import pandas as pd
import xarray as xr
import numpy as np
import shutil

# Count downloads per grid cell

data_path = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH')
data_files = [f for f in os.listdir(data_path) if f.endswith('zarr')]
df_zarr = pd.DataFrame(data_files, columns=['file'])

def extract_minx_maxy(file):
    parts = file.split('_')
    minx = int(parts[1])
    maxy = int(parts[2])
    yr = int(parts[3][:4])
    return minx, maxy, yr

# Find files where more than 2 per year and coord

df_zarr[['minx', 'maxy', 'yr']] = df_zarr['file'].apply(lambda x: pd.Series(extract_minx_maxy(x)))
grouped = df_zarr.groupby(['minx', 'maxy', 'yr']).size().reset_index(name='count')
filtered_groups = grouped[grouped['count'] > 1]
result_df = df_zarr.merge(filtered_groups, on=['minx', 'maxy', 'yr'])
result_df = result_df.drop(columns=['count'])

print(result_df)

# Find name of file to drop: the one with the shortest dates

def extract_dates(file):
    parts = file.split('_')
    startdate = int(parts[3])
    enddate = int(parts[4].split('.')[0])
    return startdate, enddate

result_df[['startdate', 'enddate']] = result_df['file'].apply(lambda x: pd.Series(extract_dates(x)))

max_enddate_indices = result_df.groupby(['minx', 'maxy', 'yr'])['enddate'].idxmax()
files_not_max_enddate = result_df[~result_df.index.isin(max_enddate_indices)]

files_max_enddate = result_df[result_df.index.isin(max_enddate_indices)]
times = []
data_path = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/')

for f in files_not_max_enddate.file:
  timelen = len(xr.open_dataset(data_path + f, engine='zarr').compute().time)
  times += [timelen]

files_not_max_enddate['times'] = times

to_del = [os.path.join(data_path, f) for f in files_not_max_enddate.file]
print(to_del)
"""
# Iterate over the list and delete each file
for file_path in to_del:
    try:
        shutil.rmtree(file_path)
        print(f"Deleted: {file_path}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")
"""


