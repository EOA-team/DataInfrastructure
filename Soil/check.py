import os
import xarray as xr
import rioxarray

data_dir =os.path.expanduser('~/mnt/eo-nas1/data/soil/ccsols/datacubes')

for i, f in enumerate(os.listdir(data_dir)): 
  if 'ccsols_612060_5160620.zarr' in f:
    print(f)
    ds = xr.open_zarr(os.path.join(data_dir, f))
    print(ds.compute())
    """
    ds = ds.assign_coords({
          "lon": ds.lon + 5,
          "lat": ds.lat - 5
      })
    ds.rename({'lat':'y', 'lon':'x'}).rio.write_crs(32632).rio.to_raster('checkfinal.tif')
    """