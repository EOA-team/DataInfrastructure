"""
Script to download tiles of SwissImage from SwissTopo.
URLs are collectd from the following website: https://www.swisstopo.admin.ch/en/orthoimage-swissimage-10#SWISSIMAGE-10-cm---Download

26 Aug. 2024
Selene Ledain
"""

import requests
import pandas as pd
import os


def si_download(urls_path, downloads_path):
    """Download from a list of urls

    Args:
        urls_path (str): path to text file containing URLS to download
        downloads_path (str): path to store downloaded files
    """

    if downloads_path.startswith('~'):
      downloads_path = os.path.expanduser(downloads_path)

    data = pd.read_csv(urls_path, header=None)
    data.columns = ["link"]
    
    for idx, row in data.iterrows():
        url = row["link"]
        name = url.split('/')[-1].split('\n')[0]
        print(name)
        file_path = os.path.join(downloads_path,name)
        
        if not os.path.exists(file_path):
          try:
            r = requests.get(url)
            open(file_path, "wb").write(r.content)
            print('Download successful')
          except requests.exceptions.RequestException as e:
            print(f'Failed to download {url}: {e}')
        else:
          continue
    
if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--urls_path', type=str, default='ch.swisstopo.swissimage-dop10-vWuyN4vG.csv')
    parser.add_argument('--downloads_path', type=str, default='~/mnt/eo-nas1/data/swisstopo/SwissImage/')

    args = parser.parse_args()
    
    si_download(args.urls_path, args.downloads_path)