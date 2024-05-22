""" 
Create a regular grid on which data should be downloaded.
The grid is aligned to pixels of the Sentinel-2 data in zone UTM 32.
The extent of the grid is defined by MeteoSuisse data, that goes slightly beyond the Swiss borders. 
We keep the Sentinel-2 pixels that cover the MeteoSuisse data.
The grid has a patch size of 1280x1280 meters. 

Sélène Ledain
May 17th 2024
"""

import geopandas as gpd

grid_path = '~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/gridface_s2tiles.shp'
gdf = gpd.read_file(grid_path)

# Keep only the patches in Switzerland

meteo_vector = gpd.read_file('~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/meteofile_vectorised.shp')
intersection = gpd.sjoin(gdf, meteo_vector, how="inner", op="intersects")
result = gdf[gdf.index.isin(intersection.index)]
result.to_file('~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/gridface_s2tiles_CH.shp')
