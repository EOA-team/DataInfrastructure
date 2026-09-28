#!/usr/bin/env python3
"""
build_scene_tier_lookup.py  [run in minicuber environment]

Queries Planetary Computer directly for all LS4/5/7 scenes over Switzerland
(1990-2025) and saves collection_category (T1/T2) per scene_id to a CSV.

Output: storage/scene_tier_lookup.csv
"""

from pathlib import Path
import pandas as pd
import pystac_client
import planetary_computer

OUT_CSV = Path(__file__).parent / "results/figures/tile_debug/scene_tier_lookup.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

CH_BBOX   = [5.0, 45.0, 11.0, 48.5]
PLATFORMS = {"landsat-4", "landsat-5", "landsat-7", "landsat-8", "landsat-9"}
YEARS     = range(1984, 2026)

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

records = []
for year in YEARS:
    print(f"  {year}...", end=" ", flush=True)
    try:
        results = catalog.search(
            collections=["landsat-c2-l2"],
            datetime=f"{year}-01-01/{year}-12-31",
            bbox=CH_BBOX,
        )
        n = 0
        for item in results.items():
            props = item.properties
            if props.get("platform") not in PLATFORMS:
                continue
            records.append({
                "scene_id":            props.get("landsat:scene_id", item.id),
                "collection_category": props.get("landsat:collection_category", "?"),
                "platform":            props.get("platform"),
                "datetime":            props.get("datetime", "")[:10],
                "wrs_path":            props.get("landsat:wrs_path"),
                "wrs_row":             props.get("landsat:wrs_row"),
                "cloud_cover":         props.get("eo:cloud_cover"),
            })
            n += 1
        print(f"{n} scenes")
    except Exception as e:
        print(f"ERROR: {e}")

df = pd.DataFrame(records).drop_duplicates("scene_id").sort_values(["platform", "datetime"])
df.to_csv(OUT_CSV, index=False)

print(f"\nSaved {len(df)} records → {OUT_CSV}")
print(df.groupby(["platform", "collection_category"]).size().to_string())
print("\nDone.")
