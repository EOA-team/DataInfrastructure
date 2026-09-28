import xarray as xr
import rioxarray 
from affine import Affine

data_path = '~/mnt/eo-nas1/data/meteo/MeteoSwiss_RhiresD_262620_5122220_20010101_20011231.zarr'
ds = xr.open_zarr(data_path)
transform = ds.rio.transform()
new_transform = Affine(transform[0], transform[1], transform[2] + 5,
                       transform[3], transform[4], transform[5] + 5)
ds.RhiresD.attrs.pop('grid_mapping')
print(ds.rio.grid_mapping) # default is spatial_ref
""" 
ds.coords['spatial_ref'] = xr.Variable((), 0)
grid_map_attrs = ds.coords['spatial_ref'].attrs.copy()
grid_map_attrs["GeoTransform"] = " ".join(
            [str(item) for item in new_transform.to_gdal()]
        )
ds.coords['spatial_ref'].rio.set_attrs(grid_map_attrs, inplace=True)
print(ds.coords['spatial_ref'].attrs)
"""
ds.rio.write_transform(new_transform, inplace=True) #for some reason only writes it Geotransfrom in attrss
print(ds.rio.transform())
print(ds.coords['spatial_ref'].attrs['GeoTransform']) 
ds.RhiresD.rio.to_raster('meteo_test.tif')

