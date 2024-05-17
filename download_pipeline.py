import os
from pathlib import Path
import sys
base_dir = Path(os.path.dirname(os.path.realpath("__file__"))).parent
sys.path.insert(0, str(base_dir))
import earthnet_minicuber as emc

import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import cascaded_union
import matplotlib.pyplot as plt
import contextily as cx
import numpy as np



def create_max_square(patch, grid, num_cells, patch_size):
    """
    Use patch as upper left corner, and create biggest possible square.
    Max side length possible is num_cells*patch size.

    :param patch: Polygon (upper left corner polygon of square to create)
    :param grid: geodataframe with other polygons that can be used
    :param num_cells: max number of cells
    :param patch_size: side of a single patch
    :return: max_square
    """

    n_cells = 0

    x, y = patch.exterior.coords.xy
    x_min, x_max, y_min, y_max = min(x), max(x), min(y), max(y)
    max_square = patch

    while n_cells <= num_cells: # Start with only patch, then adding cells
        # Calculate the coordinates of the square
        square = [
            (x_min, y_max),
            (x_max + patch_size * (n_cells), y_max),
            (x_max + patch_size * (n_cells), y_min - patch_size * (n_cells)),
            (x_min, y_min - patch_size * (n_cells)),
            (x_min, y_max)  # Close the polygon by repeating the first point
        ]
        
        # Check if this square is possible 
        square = Polygon(square)
        square_in_grid = grid[(grid.geometry.within(square)) & (~grid['selected'])]
        
        if len(square_in_grid) < (n_cells+1)**2: # the square has side ncells+1
            #print('Max found side', n_cells**2)
            return max_square
        else:
            max_square = square_in_grid
            n_cells += 1
    
    return max_square



grid_path = '~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/gridface_s2tiles_CH.shp'
grid = gpd.read_file(grid_path)

patch_size = 1280 # meters
num_cells = 8

grid['selected'] = [False]*len(grid)
grid_copy = grid.copy()
# Optional: could reload a saved grid_copy if there is

for i, row in grid.iterrows():
    if not grid_copy.loc[i, 'selected']:
        patch = row.geometry
        
        # Add surrounding patches to create up to 8x8 mega-patch (use patch as upper left corner)
        mega_patch = create_max_square(patch, grid_copy, num_cells, patch_size)
    
        # Mark the selected cells
        grid_copy.loc[mega_patch.index, 'selected'] = True

        # Optional: save grid_copy in_case need to restart loop
      

# Check that all patches were treated
any_false_selected = any(grid_copy['selected'] == False)

if any_false_selected:
    print("There are False values in the 'selected' column.")
else:
    print("All values in the 'selected' column are True.")