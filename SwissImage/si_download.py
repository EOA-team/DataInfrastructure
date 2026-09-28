"""
Script to download tiles of SwissImage from SwissTopo.
URLs are collected from the following website: https://www.swisstopo.admin.ch/en/orthoimage-swissimage-10#SWISSIMAGE-10-cm---Download

26 Aug. 2024  Selene Ledain
May 2026      Updated: timeout, retry, parallel downloads, in-memory writes
"""

import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def _make_session():
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _download_one(url, downloads_path, session, dry_run=False):
    name = url.split('/')[-1].split('\n')[0]
    file_path = os.path.join(downloads_path, name)

    if os.path.exists(file_path):
        return name, 'skipped'

    if dry_run:
        return name, f'DRY-RUN -> {file_path}'

    try:
        with session.get(url, timeout=60) as r:
            r.raise_for_status()
            content = r.content  # socket closes here when exiting the with block
        with open(file_path, 'wb') as f:
            f.write(content)
        return name, 'ok'
    except Exception as e:
        return name, f'FAILED: {e}'


def si_download(urls_path, downloads_path, workers=8, dry_run=False):
    """Download SwissImage tiles from a list of URLs.

    Args:
        urls_path (str): path to CSV file containing one URL per row
        downloads_path (str): directory to store downloaded files
        workers (int): number of parallel download threads
        dry_run (bool): if True, print what would be downloaded without writing
    """
    urls_path = os.path.expanduser(urls_path)
    downloads_path = os.path.expanduser(downloads_path)
    os.makedirs(downloads_path, exist_ok=True)

    data = pd.read_csv(urls_path, header=None)
    data.columns = ["link"]
    urls = data["link"].tolist()

    if dry_run:
        print("*** DRY RUN — no files will be written ***")
    print(f"Total tiles: {len(urls)}  |  workers: {workers}")

    session = _make_session()
    failed = []

    futures_map = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for url in urls:
            future = executor.submit(_download_one, url, downloads_path, session, dry_run)
            futures_map[future] = url

        iterable = as_completed(futures_map)
        if HAS_TQDM:
            iterable = tqdm(iterable, total=len(futures_map), unit='tile')

        for future in iterable:
            name, status = future.result()
            if dry_run or status.startswith('FAILED'):
                print(f'  {name}: {status}')
            if status.startswith('FAILED'):
                failed.append(futures_map[future])

    print(f"\nDone. {len(failed)} failed downloads.")
    if failed:
        print("Failed URLs:")
        for u in failed:
            print(f"  {u}")


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--urls_path', type=str, required=True,
                        help='Path to CSV with one download URL per row')
    parser.add_argument('--downloads_path', type=str,
                        default='/mnt/eo-nas1/data/swisstopo/SwissImage/raw/10cm/')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel download threads')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be downloaded without writing any files')

    args = parser.parse_args()

    si_download(args.urls_path, args.downloads_path, args.workers, args.dry_run)
