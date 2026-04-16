import pystac_client
import rasterio
import xarray as xr
import rioxarray
import requests
from io import BytesIO
import os


def download_SRC_from_STAC(bbox, assets, download_dir, STAC_URL='https://geoservice.dlr.de/eoc/ogc/stac/v1', collection_id="S2-soilsuite-europe-2018-2022-P5Y"):
  """ 
  Get all Soil Reflectance Composite data from the STAC API for Switzerland
  More info: https://download.geoservice.dlr.de/SOILSUITE/files/EUROPE_5Y/000_Data_Overview/SoilSuite_Data_Description_Europe_V1.pdf

  :param STAC_URL: str, STAC API endpoint (defualt https://geoservice.dlr.de/eoc/ogc/stac/v1)
  :param collection_id: str, collection ID in the STAC ("S2-soilsuite-europe-2018-2022-P5Y")
  :param bbox: list, bounding box coordinates [minlon, minlat, maxlon, maxlat] in EPSG:4326
  :param assets: list of str, choose among ['metadata', 'MREF', 'MREF-STD', 'SRC', 'SRC-STD', 'SRC-CI95', 'SFREQ', 'MASK', 'thumbnail']
  :param download_dir: where zarr files should be stored
  """
 
  # Connect to STAC catalog
  catalog = pystac_client.Client.open(STAC_URL)

  # List available collections
  #collections = [col.id for col in catalog.get_all_collections()]
  #print("Available Collections:", collections)

  # Search for items in the collection
  search = catalog.search(
      collections=[collection_id],
      bbox=bbox
  )

  items = search.item_collection()

  if not items:
    print("No matching items found.")
    return

  # Collect URLs for requested assets
  asset_urls = {asset: [] for asset in assets}  # Dictionary to store asset URLs
  for item in items:
      for asset in assets:
          if asset in item.assets:
              asset_urls[asset].append(item.assets[asset].href)

  # Download TIF files (SRC is multiband so cannot with stackstac directly)
  # Store tile by tile, creating one dataset with all different assets
  for item in items:
    print(f"Processing item: {item.id}")

    # Create a list to collect datasets for this item
    item_ds_list = []

    for asset in assets:  # Loop through each asset you're interested in
        if asset in item.assets:
            url = item.assets[asset].href  # Get the URL for the asset
            print(f"Downloading {asset} from {url}")

            # Stream file into memory
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with rasterio.open(BytesIO(response.content)) as src:
                    ds = rioxarray.open_rasterio(BytesIO(response.content))

                    if src.count == 1:  
                      # Single-band asset
                      ds = ds.squeeze().rename(asset)
                    else:  
                      long_names = ds.attrs.get('long_name', []) # the band names are available in the attributes
                      ds = ds.to_dataset("band")
                      if long_names:
                          band_names = {i+1: f"{asset}_{name.split(' ')[0]}" for i, name in enumerate(long_names)}
                          ds = ds.rename(band_names)

                    item_ds_list.append(ds)
            else:
              print(f"Failed to download {url}")

    # After processing all assets for this item, combine them into a single dataset
    if item_ds_list:
        item_ds = xr.merge(item_ds_list)  # Merge all assets into one dataset
        print(item_ds)

        # Save the dataset to disk
        tile_id = item.id.split('_')[2]
        file_path = os.path.join(download_dir, f"soilsuite_{tile_id}.zarr")
        item_ds.to_zarr(file_path, mode="w")
        print(f'Saved to {file_path}')


        



######################
# Download baresoil composite from DLR

# Will keep data in its native projection (EPSG:3035), grid and tile format provided by DLR
# There are 4 tiles covering CH: 0040-0024, 0040-0026, 0042-0024, 0042-0026

STAC_URL = "https://geoservice.dlr.de/eoc/ogc/stac/v1/"
collection_id = "S2-soilsuite-europe-2018-2022-P5Y"
bbox = [22.357, 41.235, 28.597, 44.216] # Bulgaria
""" 
[5.96, 45.82, 10.49, 47.81]  # Switzerland
[6.627, 35.288, 18.784, 47.092] # Italy
[14.123, 49.002, 24.145, 54.839] # Poland
"""
assets = ['SRC', 'SRC-STD', 'SRC-CI95', 'MASK']
download_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/raw/DLR_soilsuite/')

download_SRC_from_STAC(STAC_URL=STAC_URL, collection_id=collection_id, bbox=bbox, assets=assets, download_dir=download_dir)