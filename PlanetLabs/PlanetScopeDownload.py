import os
import json
import requests
import time
from requests.auth import HTTPBasicAuth
from pathlib import Path
import geopandas as gpd
from dotenv import load_dotenv
import pandas as pd


def download_planet(geojson_path, date_start, date_end, max_cloud_cover, API_KEY, output_path):
    # ===========================
    # Prepare Inputs
    # ===========================
    # Load AOI from GeoJSON
    if not os.path.exists(geojson_path):
        raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")

    geojson_data = gpd.read_file(geojson_path)
    if geojson_data.empty:
        raise ValueError("GeoJSON file is empty or invalid.")
    geojson_geometry = geojson_data.union_all().__geo_interface__

    # Prepare search filters
    geometry_filter = {
        "type": "GeometryFilter",
        "field_name": "geometry",
        "config": geojson_geometry
    }
    date_range_filter = {
        "type": "DateRangeFilter",
        "field_name": "acquired",
        "config": {
            "gte": f"{date_start}T00:00:00.000Z",
            "lte": f"{date_end}T23:59:59.000Z"
        }
    }
    cloud_cover_filter = {
        "type": "RangeFilter",
        "field_name": "cloud_cover",
        "config": {"lte": max_cloud_cover}
    }

    combined_filter = {
        "type": "AndFilter",
        "config": [geometry_filter, date_range_filter, cloud_cover_filter]
    }


    # ===========================
    # Search for Images
    # ===========================
    item_type = "PSScene"
    search_request = {"item_types": [item_type], "filter": combined_filter}

    response = requests.post(
        'https://api.planet.com/data/v1/quick-search',
        auth=HTTPBasicAuth(API_KEY, ''),
        json=search_request
    )

    if response.status_code != 200:
        raise Exception(f"Search request failed: {response.content}")
    data = response.json()

    # Extract Image IDs
    image_ids = [feature['id'] for feature in data.get('features', [])]
    if not image_ids:
        raise ValueError("No images found matching the criteria.")
    print(f"Found images: {image_ids}")
    
    # ===========================
    # Place the Order
    # ===========================
    order_name = f"{geojson_path}_{date_start}"
    clip_tool = {"clip": {"aoi": geojson_geometry}}
    order_request = {
        "name": order_name,
        "source_type": "scenes",
        "products": [
            {
                "item_ids": image_ids,
                "item_type": item_type,
                "product_bundle": "analytic_8b_sr_udm2"
            }
        ],
        "tools": [clip_tool]
    }

    order_url = 'https://api.planet.com/compute/ops/orders/v2'
    order_response = requests.post(
        order_url,
        auth=HTTPBasicAuth(API_KEY, ''),
        json=order_request
    )

    if order_response.status_code != 202:
        raise Exception(f"Order request failed: {order_response.content}")
    order_id = order_response.json()['id']
    print(f"Order placed successfully. Order ID: {order_id}")

    # ===========================
    # Poll Order Status
    # ===========================
    order_status_url = f"{order_url}/{order_id}"
    order_status = 'running'

    while order_status not in ['success', 'failed']:
        status_response = requests.get(order_status_url, auth=HTTPBasicAuth(API_KEY, ''))
        order_status = status_response.json()['state']
        print(f"Order status: {order_status}")
        time.sleep(10)

    if order_status != 'success':
        raise Exception("Order processing failed.")
        
    # ===========================
    # Poll Order Status & Download Files
    # ===========================
    if order_status == 'success':
        response_json = status_response.json()
        print("Order completed successfully.")
        print("Full response:", json.dumps(response_json, indent=4))  # Debugging step

        # Fetch results from response
        results = response_json.get('results') or response_json.get('_links', {}).get('results', [])
        
        # Ensure results are defined
        if not results:
            print("No downloadable results found in the response.")
            raise Exception("No results available for download.")

        
        for result in results:
            download_url = result.get('location')
            if not download_url:
                print(f"Missing 'location' in result: {result}")
                continue
        
            # Extract only the file name
            file_name = Path(result.get('name', download_url.split('/')[-1].split('?')[0])).name
            acquired_date = file_name.split('_')[0]
            save_dir = Path(output_path) / acquired_date
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            save_path = save_dir / file_name

            # Check if file already exists
            if save_path.exists():
                print(f"File {file_name} already exists. Skipping download.")
                continue

            print(f"Downloading {file_name}...")
            with requests.get(download_url, auth=HTTPBasicAuth(API_KEY, ''), stream=True) as r:
                r.raise_for_status()
                with open(save_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"File saved to {save_path}")
    
        print("All files downloaded successfully.")
    else:
        print("Order did not complete successfully; no files were downloaded.")
    
    
    return






# ===========================
# Parameters to Set
# ===========================
geojson_path = os.path.expanduser("~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/DataInfrastructure/PlanetLabs/geoms/Tänikon.geojson")
output_path = os.path.expanduser("~/mnt/eo-nas1/data/satellite/PlanetLabs/raw/Tänikon")  # Path to save output files
date_start = "2023-01-01"
date_end = "2025-12-31"   
max_cloud_cover = 0.6       # Maximum cloud cover (0-1)

load_dotenv()  # Load API key from .env into environment
API_KEY = os.getenv("API_KEY_PLANET")

years = pd.date_range(start=date_start, end=date_end, freq='YS')  # YS = Year Start

for year_start in years:
    year_end = pd.Timestamp(min(year_start.replace(month=12, day=31), pd.Timestamp(date_end)))
    print(f"Download from {year_start.date()} to {year_end.date()}")

    try:
        download_planet(geojson_path, year_start.date(), year_end.date(), max_cloud_cover, API_KEY, output_path)
    except Exception as e:
        print(e)

    

    