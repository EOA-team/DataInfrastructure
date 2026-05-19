"""
Predict SNAP LAI on all pixels of Sentinel-2 zarr cubes.

Reads cubes from
    ~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_<minx>_<maxy>_<YYYYMMDD>_<YYYYMMDD>.zarr
and writes one LAI cube per input cube to
    <output_folder>/LAI_<minx>_<maxy>_<YYYYMMDD>_<YYYYMMDD>.zarr

The LAI model is the ESA SNAP S2 toolbox v2.1 LAI neural network, implemented
in vectorised numpy (port of EOA-team/SALI_models snap_lai.py — the canonical
reference). Inputs: B03, B04, B05, B06, B07, B8A, B11, B12 (TOC reflectance,
unitless 0-1) and view_zenith, sun_zenith, relative_azimuth in degrees.

Cube conventions:
- band variables named s2_B02 ... s2_B12, scaled by 10000 (65535 = nodata)
- angle variables: mean_sensor_zenith, mean_solar_zenith, mean_sensor_azimuth,
  mean_solar_azimuth (degrees)
- product_uri stored as a per-time coordinate (carried through, unused here)
- spatial dims are (y, x) in EPSG:32632, top-left of cube at (minx, maxy)

No cloud/shadow/snow masking is applied here — every pixel gets a prediction.
NaNs only appear where input bands were already nodata (65535).

Usage:
    python predict_snap_lai.py                  # process everything missing
    python predict_snap_lai.py --workers 8      # parallel
    python predict_snap_lai.py --overwrite      # redo even if output exists
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import xarray as xr
import zarr


# ---------------------------------------------------------------------------
# SNAP LAI v2.1 — vectorised numpy port (canonical SALI_models implementation)
# ---------------------------------------------------------------------------

DEG_TO_RAD = np.pi / 180.0

# Per-input normalization ranges (from SNAP v2.1 auxdata)
_NORM = {
    "B03": (0.0, 0.253061520471542),
    "B04": (0.0, 0.290393577911328),
    "B05": (0.0, 0.305398915248555),
    "B06": (0.006637972542253, 0.608900395797889),
    "B07": (0.013972727018939, 0.753827384322927),
    "B8A": (0.026690138082061, 0.782011770669178),
    "B11": (0.016388074192258, 0.493761397883092),
    "B12": (0.0, 0.493025984460231),
    "viewZen": (0.918595400582046, 1.0),       # applied to cos(viewZen)
    "sunZen":  (0.342022871159208, 0.936206429175402),  # applied to cos(sunZen)
}

_DENORM_LAI = (0.000319182538301, 14.4675094548151)


def _normalize(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return 2.0 * (x - lo) / (hi - lo) - 1.0


def _denormalize(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return 0.5 * (x + 1.0) * (hi - lo) + lo


def predict_lai_array(
    b03: np.ndarray, b04: np.ndarray, b05: np.ndarray, b06: np.ndarray,
    b07: np.ndarray, b8a: np.ndarray, b11: np.ndarray, b12: np.ndarray,
    view_zen_deg: np.ndarray, sun_zen_deg: np.ndarray, rel_azim_deg: np.ndarray,
) -> np.ndarray:
    """Predict SNAP LAI from arrays of any (broadcasting-compatible) shape.

    Bands are TOC reflectance in [0, 1]. Angles are in degrees.
    Returns LAI with the broadcast shape; NaNs in the inputs propagate to NaNs.
    """
    b03n = _normalize(b03, *_NORM["B03"])
    b04n = _normalize(b04, *_NORM["B04"])
    b05n = _normalize(b05, *_NORM["B05"])
    b06n = _normalize(b06, *_NORM["B06"])
    b07n = _normalize(b07, *_NORM["B07"])
    b8an = _normalize(b8a, *_NORM["B8A"])
    b11n = _normalize(b11, *_NORM["B11"])
    b12n = _normalize(b12, *_NORM["B12"])

    vz = _normalize(np.cos(view_zen_deg * DEG_TO_RAD), *_NORM["viewZen"])
    sz = _normalize(np.cos(sun_zen_deg  * DEG_TO_RAD), *_NORM["sunZen"])
    ra = np.cos(rel_azim_deg * DEG_TO_RAD)  # already in [-1, 1]

    # Layer 1: 5 tanh neurons
    n1 = np.tanh(
        4.96238030555279
        - 0.023406878966470 * b03n + 0.921655164636366 * b04n
        + 0.135576544080099 * b05n - 1.938331472397950 * b06n
        - 3.342495816122680 * b07n + 0.902277648009576 * b8an
        + 0.205363538258614 * b11n - 0.040607844721716 * b12n
        - 0.083196409727092 * vz   + 0.260029270773809 * sz
        + 0.284761567218845 * ra
    )
    n2 = np.tanh(
        1.416008443981500
        - 0.132555480856684 * b03n - 0.139574837333540 * b04n
        - 1.014606016898920 * b05n - 1.330890038649270 * b06n
        + 0.031730624503341 * b07n - 1.433583541317050 * b8an
        - 0.959637898574699 * b11n + 1.133115706551000 * b12n
        + 0.216603876541632 * vz   + 0.410652303762839 * sz
        + 0.064760155543506 * ra
    )
    n3 = np.tanh(
        1.075897047213310
        + 0.086015977724868 * b03n + 0.616648776881434 * b04n
        + 0.678003876446556 * b05n + 0.141102398644968 * b06n
        - 0.096682206883546 * b07n - 1.128832638862200 * b8an
        + 0.302189102741375 * b11n + 0.434494937299725 * b12n
        - 0.021903699490589 * vz   - 0.228492476802263 * sz
        - 0.039460537589826 * ra
    )
    n4 = np.tanh(
        1.533988264655420
        - 0.109366593670404 * b03n - 0.071046262972729 * b04n
        + 0.064582411478320 * b05n + 2.906325236823160 * b06n
        - 0.673873108979163 * b07n - 3.838051868280840 * b8an
        + 1.695979344531530 * b11n + 0.046950296081713 * b12n
        - 0.049709652688365 * vz   + 0.021829545430994 * sz
        + 0.057483827104091 * ra
    )
    n5 = np.tanh(
        3.024115930757230
        - 0.089939416159969 * b03n + 0.175395483106147 * b04n
        - 0.081847329172620 * b05n + 2.219895367487790 * b06n
        + 1.713873975136850 * b07n + 0.713069186099534 * b8an
        + 0.138970813499201 * b11n - 0.060771761518025 * b12n
        + 0.124263341255473 * vz   + 0.210086140404351 * sz
        - 0.183878138700341 * ra
    )

    # Layer 2: linear
    out = (
        1.096963107077220
        - 1.500135489728730 * n1 - 0.096283269121503 * n2
        - 0.194935930577094 * n3 - 0.352305895755591 * n4
        + 0.075107415847473 * n5
    )

    return _denormalize(out, *_DENORM_LAI)


# ---------------------------------------------------------------------------
# Cube I/O
# ---------------------------------------------------------------------------

NODATA_RAW = 65535            # sentinel value in raw bands
REFLECTANCE_SCALE = 10000.0   # raw S2 -> reflectance in [0, 1]

# Bands consumed by the SNAP LAI model, in the order it expects them.
LAI_BANDS = ["s2_B03", "s2_B04", "s2_B05", "s2_B06",
             "s2_B07", "s2_B8A", "s2_B11", "s2_B12"]


def parse_cube_filename(name: str) -> dict:
    """Parse 'S2_<minx>_<maxy>_<YYYYMMDD>_<YYYYMMDD>.zarr' into its parts."""
    stem = name[:-len(".zarr")] if name.endswith(".zarr") else name
    parts = stem.split("_")
    if len(parts) != 5 or parts[0] != "S2":
        raise ValueError(f"Unexpected cube filename: {name}")
    return {"minx": int(parts[1]), "maxy": int(parts[2]),
            "start": parts[3],     "end":  parts[4]}


def output_filename(cube_name: str) -> str:
    """Output name mirrors the input, with 'S2' replaced by 'LAI'."""
    p = parse_cube_filename(cube_name)
    return f"LAI_{p['minx']}_{p['maxy']}_{p['start']}_{p['end']}.zarr"


def list_cubes(data_folder: str) -> list[str]:
    """All S2_*.zarr cubes in `data_folder`, sorted alphabetically."""
    return sorted(
        f for f in os.listdir(data_folder)
        if f.startswith("S2_") and f.endswith(".zarr")
    )


def load_reflectance(ds: xr.Dataset) -> dict[str, xr.DataArray]:
    """Convert raw S2 bands to float32 reflectance, masking the 65535 nodata.

    Returns a {band_name: DataArray} dict for the bands SNAP LAI needs.
    """
    bands = {}
    for var in LAI_BANDS:
        if var not in ds:
            raise KeyError(f"missing band variable {var}")
        raw = ds[var]
        bands[var] = (
            raw.where(raw != NODATA_RAW).astype("float32") / REFLECTANCE_SCALE
        )
    return bands


def load_angles(ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Return (view_zenith, sun_zenith, relative_azimuth) in degrees.

    Relative azimuth is computed as (sun_azimuth - view_azimuth) when both are
    present; otherwise a stored 'relative_azimuth' is used.
    """
    for var in ("mean_sensor_zenith", "mean_solar_zenith"):
        if var not in ds:
            raise KeyError(f"missing angle variable {var}")

    view_zen = ds["mean_sensor_zenith"].astype("float32")
    sun_zen  = ds["mean_solar_zenith"].astype("float32")

    if "mean_solar_azimuth" in ds and "mean_sensor_azimuth" in ds:
        rel_azim = (ds["mean_solar_azimuth"] - ds["mean_sensor_azimuth"]).astype("float32")
    elif "relative_azimuth" in ds:
        rel_azim = ds["relative_azimuth"].astype("float32")
    else:
        raise KeyError(
            "missing relative azimuth: need either "
            "(mean_solar_azimuth, mean_sensor_azimuth) or 'relative_azimuth'"
        )

    return view_zen, sun_zen, rel_azim


def predict_lai_dataset(ds: xr.Dataset) -> xr.DataArray:
    """Run the SNAP LAI model on a full xarray cube and return a DataArray."""
    bands = load_reflectance(ds)
    view_zen, sun_zen, rel_azim = load_angles(ds)

    # Angles may be per-time scalars; broadcast them to the (time, y, x) grid.
    target = bands["s2_B03"]
    view_zen, _ = xr.broadcast(view_zen, target)
    sun_zen,  _ = xr.broadcast(sun_zen,  target)
    rel_azim, _ = xr.broadcast(rel_azim, target)

    lai = xr.apply_ufunc(
        predict_lai_array,
        bands["s2_B03"], bands["s2_B04"], bands["s2_B05"], bands["s2_B06"],
        bands["s2_B07"], bands["s2_B8A"], bands["s2_B11"], bands["s2_B12"],
        view_zen, sun_zen, rel_azim,
        dask="parallelized",
        output_dtypes=[np.float32],
    )
    lai.name = "lai"
    lai.attrs.update({
        "long_name": "Leaf Area Index (SNAP S2 v2.1)",
        "units": "m2/m2",
        "valid_min": 0.0,
        "valid_max": 15.0,
        "source": "ESA SNAP S2 toolbox biophysical processor v2.1",
    })
    return lai


def write_lai_cube(lai: xr.DataArray, source_ds: xr.Dataset,
                   source_name: str, output_path: str) -> None:
    """Wrap LAI into a Dataset, copy useful metadata, and write to zarr."""
    out = xr.Dataset({"lai": lai})

    if "product_uri" in source_ds.coords or "product_uri" in source_ds.data_vars:
        out = out.assign_coords(product_uri=source_ds["product_uri"].astype(str))

    for key in ("crs", "spatial_ref", "projection"):
        if key in source_ds.attrs:
            out.attrs[key] = source_ds.attrs[key]
    out.attrs.setdefault("crs", "EPSG:32632")
    out.attrs["source_cube"] = source_name

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
    out.to_zarr(
        output_path, mode="w", consolidated=True,
        encoding={"lai": {"compressor": compressor, "dtype": "float32"}},
    )


def process_cube(cube_path: str, output_path: str) -> None:
    """End-to-end: open one cube, predict LAI, write the result."""
    ds = xr.open_zarr(cube_path)
    lai = predict_lai_dataset(ds)
    write_lai_cube(lai, ds, os.path.basename(cube_path), output_path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def process_one(cube_name: str, data_folder: str, output_folder: str,
                overwrite: bool) -> tuple[str, bool, str]:
    """Worker: process one cube. Returns (name, ok, message)."""
    out_name = output_filename(cube_name)
    out_path = os.path.join(output_folder, out_name)

    if os.path.exists(out_path) and not overwrite:
        return cube_name, True, "skipped (exists)"

    try:
        process_cube(os.path.join(data_folder, cube_name), out_path)
        return cube_name, True, f"-> {out_name}"
    except Exception as e:
        return cube_name, False, f"{type(e).__name__}: {e}"


def run(data_folder: str, output_folder: str,
        workers: int = 1, overwrite: bool = False) -> None:
    """Predict LAI on every cube in `data_folder`, writing to `output_folder`."""
    os.makedirs(output_folder, exist_ok=True)
    cubes = list_cubes(data_folder)
    if not cubes:
        print(f"No cubes found in {data_folder}", file=sys.stderr)
        return

    print(f"Found {len(cubes)} cubes in {data_folder}")
    print(f"Writing LAI to {output_folder}")
    print(f"Workers: {workers}  overwrite: {overwrite}")

    work = partial(process_one,
                   data_folder=data_folder,
                   output_folder=output_folder,
                   overwrite=overwrite)

    n_ok = n_fail = 0
    if workers <= 1:
        results = (work(c) for c in cubes)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        futures = [pool.submit(work, c) for c in cubes]
        results = (f.result() for f in as_completed(futures))

    for name, ok, msg in results:
        status = "OK " if ok else "ERR"
        print(f"[{status}] {name}  {msg}")
        n_ok += int(ok)
        n_fail += int(not ok)

    if workers > 1:
        pool.shutdown()

    print(f"\nDone. {n_ok} ok, {n_fail} failed.")


def main():
    default_in  = os.path.expanduser("~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH")
    default_out = os.path.expanduser("~/mnt/eo-nas1/data/satellite/sentinel2/SNAP_LAI")

    p = argparse.ArgumentParser(description="Predict SNAP LAI on Sentinel-2 zarr cubes.")
    p.add_argument("--data-folder",   default=default_in,
                   help=f"Folder of S2_*.zarr cubes (default: {default_in})")
    p.add_argument("--output-folder", default=default_out,
                   help=f"Folder to write LAI_*.zarr (default: {default_out})")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel processes (default: 1)")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing outputs")
    args = p.parse_args()

    run(
        data_folder=args.data_folder,
        output_folder=args.output_folder,
        workers=args.workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
