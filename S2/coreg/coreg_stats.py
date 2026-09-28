import os
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import glob
from scipy.ndimage import affine_transform
import matplotlib.pyplot as plt
import zarr
import datetime
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from scipy.stats import pearsonr
from skgstat import Variogram


# Load computed shifts
shift_files = [f for f in os.listdir('.') if f.startswith('shift_results')]
files = []
for f in shift_files:
    files.append(pd.read_pickle(f))
df = pd.concat(files)

df = df.dropna(subset='uri')
df['x'] = df['name'].apply(lambda x: int(x.split('_')[0]))
df['y'] = df['name'].apply(lambda x: int(x.split('_')[1]))
df['timestamp'] = df['uri'].apply(lambda x: x.split('_')[2].split('T')[0])
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d')
df['tile'] = df['uri'].apply(lambda x: x.split('_')[5]) 
# Keep only one tile
df = df[df.tile=='T32TMT']

"""
# Count per uri
time_counts = df.groupby('timestamp').size()
plt.figure(figsize=(16, 5))
plt.plot(time_counts.index, time_counts.values, marker='o', linestyle='-', color='steelblue')
plt.xlabel('Timestamp')
plt.ylabel('Row Count')
plt.title('Number of Rows per Timestamp')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('counts_per_time.png')

# Stdev per uri across time
std_dev = df.groupby('timestamp')[['shift_x', 'shift_y']].std()
plt.figure(figsize=(16, 5))
plt.plot(std_dev.index, std_dev['shift_x'], label='std(shift_x)', marker='o')
#plt.plot(std_dev.index, std_dev['shift_y'], label='std(shift_y)', marker='s')
plt.xlabel('Timestamp')
plt.ylabel('Standard Deviation')
plt.title('Variation of Shift Standard Deviation Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.savefig('stdev_per_time.png')

# Heatmap of shifts over space (mean shift in time at each location)
heatmap_data = df.pivot_table(index='y', columns='x', values='shift_x', aggfunc='mean').dropna().apply(pd.to_numeric, errors='coerce')
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap='coolwarm', annot=False, fmt='.2f', linewidths=0.5)
plt.gca().invert_yaxis()
plt.title('Heatmap of Shift_x in Space')
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.savefig('heatmap_x.png')

heatmap_data = df.pivot_table(index='y', columns='x', values='shift_y', aggfunc='mean').dropna().apply(pd.to_numeric, errors='coerce')
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap='coolwarm', annot=False, fmt='.2f', linewidths=0.5)
plt.gca().invert_yaxis()
plt.title('Heatmap of Shift_y in Space')
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.savefig('heatmap_y.png')
""" 

# Variogram
## First, need to make sure all (x,y) have the same number of timestamps. If they are missing just add them with shifts of 0
""" 
all_times = pd.unique(df['timestamp'])
df_agg = df.drop(['uri', 'tile', 'name'], axis=1).groupby(['timestamp', 'x', 'y'], as_index=False).mean()
df_pivot_x = df_agg.pivot(index='timestamp', columns=['x', 'y'], values='shift_x').reindex(all_times).fillna(0)
df_pivot_y = df_agg.pivot(index='timestamp', columns=['x', 'y'], values='shift_y').reindex(all_times).fillna(0)
corr_matrix_x = df_pivot_x.corr()
"""

timestamps = sorted(df['timestamp'].unique())
spatial_corr_over_time = {}

# For each timestamp, compute varioagram
for i, t in enumerate(timestamps):
    print(f'Variogram for time {i}/{len(timestamps)}')
    df_time = df[df['timestamp'] == t]
    
    df_time_pivot_x = df_time.pivot_table(index='y', columns='x', values='shift_x').fillna(0)
    df_time_pivot_y = df_time.pivot_table(index='y', columns='x', values='shift_y').fillna(0)

    coords_shiftx = np.array([(x, y) for x in df_time_pivot_x.columns for y in df_time_pivot_x.index])
    coords_shifty = np.array([(x, y) for x in df_time_pivot_y.columns for y in df_time_pivot_y.index])

    shift_x_values = df_time_pivot_x.values.flatten()
    shift_y_values = df_time_pivot_y.values.flatten()

    if len(shift_x_values) < 2 or len(shift_y_values) < 2:
        print(f"Skipping timestamp {t} due to insufficient data points")
        spatial_corr_over_time[t] = {'range_x': np.nan, 'range_y': np.nan}
        continue

    # Create the variogram for 'shift_x' and 'shift_y' correlations
    variogram_x = Variogram(coords_shiftx, shift_x_values)
    variogram_y = Variogram(coords_shifty, shift_y_values)

    variogram_x.fit()
    variogram_y.fit()

    spatial_corr_over_time[t] = {
        'range_x': variogram_x.describe().get('effective_range'),
        'range_y': variogram_y.describe().get('effective_range')
    }


range_x_values = [spatial_corr_over_time[t]['range_x'] for t in timestamps]
range_y_values = [spatial_corr_over_time[t]['range_y'] for t in timestamps]


# Plot the spatial correlation ranges over time
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

axs[0].plot(timestamps, range_x_values, label='Range (shift_x)', color='blue')
axs[0].set_xlabel('Timestamp')
axs[0].set_ylabel('Range (meters)')
axs[0].set_title('Spatial Correlation Range for shift_x Over Time')

axs[1].plot(timestamps, range_y_values, label='Range (shift_y)', color='red')
axs[1].set_xlabel('Timestamp')
axs[1].set_ylabel('Range (meters)')
axs[1].set_title('Spatial Correlation Range for shift_y Over Time')

plt.tight_layout()
plt.savefig('variogram_range.png')
    

""" 
# Pearson correlation of mean shift in time per location
## First, need to make sure all (x,y) have the same number of timestmpas. If they are missing just add them with shifts of 0
all_times = pd.unique(df['timestamp'])
df_agg = df.drop(['uri', 'tile', 'name'], axis=1).groupby(['timestamp', 'x', 'y'], as_index=False).mean()
df_pivot_x = df_agg.pivot(index='timestamp', columns=['x', 'y'], values='shift_x').reindex(all_times).fillna(0)
df_pivot_y = df_agg.pivot(index='timestamp', columns=['x', 'y'], values='shift_y').reindex(all_times).fillna(0)

corr_matrix_x = df_pivot_x.corr()
corr_matrix_x.index.names = ['y_idx', 'x_idx']
corr_matrix_x.columns.names = ['y_col', 'x_col']
corr_matrix_x = corr_matrix_x.droplevel(0, axis=1)
corr_matrix_x = corr_matrix_x.droplevel(1, axis=0)
print(corr_matrix_x.columns)
print(corr_matrix_x.index)

# Plot correlation heatmaps
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

sns.heatmap(corr_matrix_x, cmap='coolwarm', annot=False, linewidths=0.5, ax=axes[0], vmin=-0.1, vmax=0.1)
axes[0].set_title('Correlation of Shift X Over Time')
axes[0].set_xlabel('Location (index)')
axes[0].set_ylabel('Location (index)')
plt.savefig('corr_mean_time.png')
plt.show()
"""


    