# Swiss Agricultural Landscape Intelligence (SALI) platform
The EOA data infrastructure. 

This repository documents how the multi-source Earth observation dataset was produced and where it lives. All data is stored on the shared NAS at `\\eo-nas1\data`.
## :ledger: Index

1. [Environment setup](#environment-setup)
2. [Grid creation](#grid-creation)
3. [Sentinel-2](#sentinel-2)
4. [MeteoSwiss](#meteoswiss)
5. [SwissImage](#swissimage)
6. [swissalti3D (DEM)](#swissalti3d)
7. [Landsat](#landsat)
8. [DLR Soilsuite](#dlr)
9. [PlanetLabs](#planet)
10. [Soil properties — ccsols](#ccsols)
11. [Snow depth](#snowdepth)
12. [LAI](#lai)
13. [Data status](#data-status)
14. [Storage structure](#storage-structure)


## 1. Environment setup <a name="environment-setup"></a>

Install with:
```bash
pip install -r requirements.txt
```
The Sentinel-2 and Landsat pipelines depend on the [minicuber](https://github.com/EOA-team/minicuber) package (commit pinned in the [Data status](#data-status) table).

---
## 2. Grid creation <a name="grid-creation"></a>

The data is saved on the EPSG:32632 grid. The pixels align to those of the Sentinel-2 satellite data in the UTM zone 32. The extent of the grid is defined by the bounding box of the MeteoSuisse data, which extends slightly beyond the administrative borders of Switzerland. 

The grid has a resolution of 1280m x 1280m, meaning 128x128 Sentinel-2 pixels (10m resolution) are included per grid cell. To create the grid, run the QGIS model in `grid_creator.model3` in QGIS. The inputs are:
- a weather file from MeteoSuisse (e.g.`O:/Data-Raw/27_Natural_Resources-RE/99_Meteo_Public/MeteoSwiss_netCDF/__griddedData/lv95updated/TminY_ch01r.swiss.lv95_202301010000_202301010000.nc`)
- an image from the Sentinel-2 tile T32TPS (e.g.`~/mnt/eo-nas1/data/satellite/sentinel2/CH_old/2020/S2A_MSIL2A_20200228T101021_N0214_R022_T32TPS_20200228T114852.SAFE/GRANULE/L2A_T32TPS_A024472_20200228T101400/IMG_DATA/R10m/T32TPS_20200228T101021_B03_10m.jp2`)

The resulting grid will start at the first pixel of T32TPS covering the weather file. This corresponds to the eastern most point of the data. The grid is then extended to cover the entire weather file. It is provided at the following path:
```
~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/gridface_s2tiles.shp
```

<p align="center">
  <img src="img/grid_all.png" width="400" height="300">
</p>
<p align="center">
    <em>Sentinel-2 aligned grid covering the bounding box of MeteoSuisse file (shown in background)</em>
</p>


Since this grid is a rectangle and contains multiple tiles outside of Switzerland, it is cropped such as to keep only the grid cells that fall over the MeteoSuisse file:
```
python crop_grid.py
```

The final grid is saved at
```
~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/gridface_s2tiles_CH.shp
```

<p align="center">
  <img src="img/grid_CH.png" width="500" height="300">
</p>
<p align="center">
    <em>Sentinel-2 aligned grid cropped to MeteoSuisse file</em>
</p>

---

## 3. Sentinel-2 <a name="sentinel-2"></a>

**Temporal coverage:** 2016-2025  
**Spatial resolution:** 10 m  
**CRS:** EPSG:32632  
**Tile size:** 128 × 128 pixels (1280 m × 1280 m)

The Sentinel-2 data is downloaded using the grid created above. Each grid tile is 1280m x 1280m, containing 128 x 128 pixels with a resolution of 10m.\
For each grid tile, the data is queried using the [minicuber](https://github.com/EOA-team/minicuber/tree/main) code which takes care of reprojecting all data to a common, 10m resolution pixel size and aligned to the coordinates of the grid (EPSG:32632). Details on other processing steps are included in the minicuber documentation.

### How to run

```bash
python S2/download_pipeline_parallel.py
```

Key parameters (hard-coded at the top of the script):
- `output_folder` — where zarr files are written
- `num_cells` — number of adjacent tiles merged into one query (reduces Microsoft Planetary Computer round-trips). A value of 2 means 3×3 tiles are fetched at once and then split. Automatically reduced at grid borders.
- Downloads run year by year; parallelisation is across years.

### Bands and variables

```
s2_AOT, s2_B01, s2_B02, s2_B03, s2_B04, s2_B05, s2_B06, s2_B07,
s2_B08, s2_B8A, s2_B09, s2_B11, s2_B12, s2_WVP, s2_SCL,
s2_mask (DL cloud mask, see minicuber docs),
product_uri, mean_sensor_zenith, mean_sensor_azimuth,
mean_solar_zenith, mean_solar_azimuth
```

### Data location and format

```
~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/
```

Files: `S2_<minx>_<maxy>_<startYYYYMMDD>_<endYYYYMMDD>.zarr`  
`(minx, maxy)` = upper-left corner in EPSG:32632. Each zarr is split into two chunks along the longitude axis.

> **Note — known data quality issues:**
> - When `num_cells > 1`, timestamps that don't span all tiles in the block are filled with NaN. These all-NaN timestamps should be dropped before use.
> - `product_uri` records the originating Sentinel-2 L2A product. Atmospheric correction was applied upstream — see [Copernicus S2 products](https://sentiwiki.copernicus.eu/web/s2-products).
> - Potential improvement: parallelise across grid tiles rather than years for future single-year updates.

---

## 4. MeteoSwiss <a name="meteoswiss"></a>

**Temporal coverage:** 1961–2025 (varies by variable)  
**Original spatial resolution:** 1 km  
**Processed resolution:** 10 m (nearest-neighbour upsampling — see warning below)  
**CRS:** EPSG:32632

### Raw data location

```
O:/Data-Raw/27_Natural_Resources-RE/99_Meteo_Public/MeteoSwiss_netCDF/__griddedData/lv95updated/
```

### How to run

```bash
python Meteo/meteo_to_cube_parallel.py
```

Reprojects the native LV95 (EPSG:2056) netCDF files to EPSG:32632 and regrid to the S2-aligned 10 m grid.

### Variables

| Variable | Description | Unit |
|----------|-------------|------|
| `Rhires` | Daily precipitation | mm |
| `Srel` | Daily relative sunshine duration | % |
| `Tabs` | Daily mean air temperature | °C |
| `Tmin` | Daily minimum air temperature | °C |
| `Tmax` | Daily maximum air temperature | °C |

For full product documentation see the [MeteoSwiss product page](https://www.meteoswiss.admin.ch/dam/jcr:215c313a-dc13-4b67-bca0-dbd966597f9a/ProdDoc_Cover-dfie.pdf).

### Data location and format

```
~/mnt/eo-nas1/data/meteo/<datavar>/MeteoSwiss_<datavar>D_<minx>_<maxy>_<year>0101_<year>1231.zarr
```

> **Warning:** Nearest-neighbour interpolation was used for upsampling from 1 km to 10 m. The data structurally resembles 1 km resolution. For true 10 m meteorological data, reprocess with bilinear interpolation.

---

## 5. SwissImage (Swisstopo) <a name="swissimage"></a>

**Spatial resolution:** 10 cm or 2 m  
**CRS (raw):** EPSG:2056 → reprojected to EPSG:32632  
**Product:** [swissimage-dop10](https://www.swisstopo.admin.ch/en/orthoimage-swissimage-10)

### Download raw TIF files

```bash
python SwissImage/si_download.py --urls_path path/to/urls.csv --downloads_path path/to/output/folder
```

URL lists:
- `SwissImage/ch.swisstopo.swissimage-dop10-DOp5jXFT.csv` — 10 cm resolution
- `SwissImage/ch.swisstopo.swissimage-dop10-vWuyN4vG.csv` — 2 m resolution

Raw TIF filename structure: `swissimage-dop10_<YEAR>_<MINX>_<MINY>_<RESOLUTION>_2056.tif`  
(MINX, MINY in km, EPSG:2056; one file = 1 km × 1 km, RGB.)

Raw files stored at:
- `~/mnt/eo-nas1/data/swisstopo/SwissImage/raw/10cm`
- `~/mnt/eo-nas1/data/swisstopo/SwissImage/raw/2m`

### Reproject and align to S2 grid

```bash
python SwissImage/si_to_cube_parallel.py
```

Reprojects to EPSG:32632 and resamples to align to the custom Sentinel-2 grid (preserving 10 cm or 2 m resolution).

### Data location and format

```
~/mnt/eo-nas1/data/swisstopo/SwissImage/cubes/10cm/
~/mnt/eo-nas1/data/swisstopo/SwissImage/cubes/2m/
```

---

## 6. swissalti3D (Swisstopo DEM) <a name="swissalti3d"></a>

**Original resolution:** 2 m  
**CRS (raw):** EPSG:2056 → reprojected to EPSG:32632  
**Product:** [swissalti3D](https://www.swisstopo.admin.ch/en/height-model-swissalti3d)

### How to run

```bash
python DEM/dem_to_cube_inverse.py
```

Reprojects from EPSG:2056 to EPSG:32632 using nearest-neighbour interpolation, aligned to the custom grid.

### Data location and format

```
~/mnt/eo-nas1/data/swisstopo/DEM/sa3D_<minx>_<maxy>.zarr
```

`(minx, maxy)` = top-left corner in EPSG:32632.

> **Warning:** The DEM source uses 65535 as a NoData value. This was not always correctly handled during reprojection, so grid tiles at the Swiss border may have artificially elevated values. These tiles most likely all fall outside the Swiss boundary. Fix: pass the correct NoData flag during reprojection.

---


## 7. Landsat <a name="landsat"></a>

**Temporal coverage:** 1982–2012 (L4-5), 2013–present (L8-9)  
**Spatial resolution:** 30 m  
**CRS:** EPSG:32632  
**Tile size:** 128 × 128 pixels (3840 m × 3840 m)

Landsat uses a **separate grid** from S2 (different pixel spacing and origin). S2 and Landsat can be reconciled by restructuring S2 data to 30 m.

### Grid creation

The grid originates from the upper-left corner of Landsat path 195 / row 27 (already in EPSG:32632). Tiles are 3840 m × 3840 m.

```bash
python landsat/landsat_grid.py   # create rectangle grid
python crop_grid.py              # crop to Switzerland
```

Final grid:
```
~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/grid_landsat_CH.shp
```

### How to run

Choose the script matching the satellite generation:

| Script | Satellites | Years |
|--------|-----------|-------|
| `landsat/download_pipeline_parallel_45.py` | Landsat 4 & 5 (TM) | 1982–2012 |
| `landsat/download_pipeline_parallel_7.py` | Landsat 7 (ETM+) | 1999–2022 |
| `landsat/download_pipeline_parallel.py` | Landsat 8 & 9 (OLI/TIRS) | 2013–present |

Key parameter to update in scripts: the target year range (hard-coded near the top of each file).

### Bands and variables

**Landsat 4–5 (TM):**
`TM_B1` (blue), `TM_B2` (green), `TM_B3` (red), `TM_B4` (NIR), `TM_B5` (SWIR1), `TM_B6` (LWIR/surface temp), `TM_B7` (SWIR2), `SR_ATMOS_OPACITY`

**Landsat 8–9 (OLI/TIRS):**
`OLI_B1` (coastal aerosol), `OLI_B2` (blue), `OLI_B3` (green), `OLI_B4` (red), `OLI_B5` (NIR), `OLI_B6` (SWIR1), `OLI_B7` (SWIR2), `TIRS_B10` (LWIR11/surface temp)

**Quality / QA bands (all generations):** `ST_QA`, `QA_RADSAT`, `QA_PIXEL`, `QA_AEROSOL`, `SR_CLOUD_QA`

**Surface temperature ancillary:** `ST_ATRAN`, `ST_DRAD`, `ST_TRAD`, `ST_URAD`, `ST_EMIS`, `ST_EMSD`, `ST_CDIST`

**Metadata:** such as scene id, original projetion, orbit path and row...

Each variable has metadata with description, scale/offset to convert to reflectance, and fill value. When raw data is in EPSG:32631 it is reprojected to EPSG:32632.

See [USGS band designations](https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites) for details.

### Data location and format

```
~/mnt/eo-nas1/data/satellite/landsat/raw/CH/45/   # Landsat 4-5
~/mnt/eo-nas1/data/satellite/landsat/raw/CH/89/   # Landsat 8-9
```

Files: `LS_<minx>_<maxy>_<startYYYYMMDD>_<endYYYYMMDD>.zarr`

> **Note:** Same NaN-timestamp issue as S2 (see section 3) may occur. Drop all-NaN timestamps before use.

---

## 8. DLR Soilsuite <a name="dlr"></a>

**Temporal coverage:** 2018–2022 composite (5-year)  
**Source:** [DLR EO Center soilsuite](https://geoservice.dlr.de/web/datasets/soilsuite_eur_5y) — bare soil composite from Sentinel-2  
**CRS:** EPSG:32632 (reprojected from source)

### How to run

```bash
python DLR_soilsuite/download_src.py    # download raw TIFs
python DLR_soilsuite/SRC_to_cube.py     # reproject to EPSG:32632 and save as zarr
```

### Data location and format

```
~/mnt/eo-nas1/data/satellite/sentinel2/DLR_soilsuite/SRC_<left>_<top>.zarr
```

`(left, top)` = top-left corner aligned to the S2 grid.

---

## 9. PlanetLabs <a name="planet"></a>

**Product:** PlanetScope 8-band (PSB.SD), TIF clipped to AOI  
**Cloud cover filter:** < 60% scene cloud coverage  
**CRS:** native PlanetScope projection (not reprojected)

AOI geometries (GeoJSON) are stored in `PlanetLabs/geoms/`. To download a new AOI, edit the AOI and date range in:

```bash
python PlanetLabs/PlanetScopeDownload.py
```

Requires a Planet API key (set in `PlanetLabs/.env` — **do not commit this file**).

---

## 10. Soil properties — ccsols <a name="ccsols"></a>

**Source:** Swiss Competence Centre for Soils (KOBO) — [ccsols](https://www.kobocat.ch)  
**Original resolution:** 30 m, EPSG:2056  
**Processed resolution:** 10 m, EPSG:32632 (nearest-neighbour interpolation)

Raw raster layers: `~/mnt/eo-nas1/data/soil/ccsols/Daten_2024-01`

### How to run

```bash
python Soil/soil_to_cube.py
```

### Variables (each at depths 0–30, 30–60, 60–120 cm)

| Variable | Description | Unit |
|----------|-------------|------|
| `CECpot_depth_*` | Cation exchange capacity | mmolc/kg |
| `clay_depth_*` | Clay content | % |
| `pH_depth_*` | pH | – |
| `sand_depth_*` | Sand content | % |
| `silt_depth_*` | Silt content | % |
| `soc_depth_*` | Soil organic carbon | % |

### Data location and format

```
~/mnt/eo-nas1/data/soil/ccsols/datacubes/ccsols_<minx>_<maxy>.zarr
```

---

## 11. Snow depth <a name="snowdepth"></a>

**Temporal coverage:** 1970s–2023 (raw), 2015–2023 (processed cubes)  
**Original resolution:** 1 km, EPSG:2056  
**Processed resolution:** 10 m, EPSG:32632 (nearest-neighbour)

Raw data (daily snow depth):
```
~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/DataInfrastructure/Meteo/HSCLQMD_ch01h.swiss.lv95_WY_1962_2023.nc
```

### How to run

```bash
python Meteo/snow.py
```

### Data location and format

```
~/mnt/eo-nas1/data/meteo/snowdepth/snowdepth_<minx>_<maxy>.zarr
```

> **Note:** Nearest-neighbour upsampling used (same caveat as MeteoSwiss — see section 5).

---


## 12. LAI <a name="lai"></a>

**Temporal coverage:** 2016-2025 (Sentinel-2)
**Processed resolution:** 10 m, EPSG:32632 

Predict LAI using ESA's SNAP Biophysical Processor on Sentinel-2 L2A data. The SNAP Biophysical Processor uses the PROSAIL radiative transfer model to estimate LAI from surface reflectance data (bands 3,4,5,6,7,8a,11,12). For full code and details on SNAP LAI please refer to the following [SNAP_LAI](https://github.com/EOA-team/SNAP_LAI/tree/main), [SALI_models](https://github.com/EOA-team/SALI_models).


### How to run

```bash
python LAI/predict_snap_lai.py
```

### Data location and format

```
~/mnt/eo-nas1/data/satellite/sentinel2/SNAP_LAI/ 
```
Each zarr file has the format `S2_<minx>_<maxy>_<startYYYYMMDD>_<endYYYYMMDD>.zarr`, keeping the exact same name as the origal S2 data cube it was predicted from. The file contains a single band called `lai`, matching the pixel coordinates of the original S2 data.

---

## Data status <a name="Data-status"></a>

The download history is tracked here:

| Date | Jobs | Notes | 
| --------- | ------------ | ------------ |
| 31.05.2024| Run S2 downloading | Package versions: sen2nbar==2023.8.1  minicuber ([commit version](https://github.com/EOA-team/minicuber/tree/14eb81ee93f91c0076e21debf23e4a82e6d7cc9e))| 
| 18.07.2024| Completed S2 download | | 
| 26.07.2024| Downloaded SwissImage 2m | | 
| 29.07.2024| Downloaded SwissImage 0.1m | |
| 09.09.2024| Processed SwissImage 0.1m and 2m to cubes | |
| 09.09.2024| Downloaded S2 2016 |Package versions: sen2nbar==2023.8.1  minicuber ([commit version](https://github.com/EOA-team/minicuber/tree/14eb81ee93f91c0076e21debf23e4a82e6d7cc9e))| 
| 25.11.2024| Processed MeteoSwiss variables to cubes | |
| 02.12.2024| Processed swissalti3D to cubes | |
| 28.02.2025| Downloaded Landsat 8-9 and processed to cubes |Package versions: sen2nbar==2023.8.1  minicuber ([commit version](https://github.com/EOA-team/minicuber/tree/3660b36fd5b61f93b9984634d95b136420f3ea3c)) |
| 20.03.2025| Downloaded S2 for 2024 | Package versions: sen2nbar==2023.8.1  minicuber ([commit version](https://github.com/EOA-team/minicuber/tree/686227f8fc0131053a448a3db59a6054e44c08da))| 
| 05.2025| Downloaded and processed DLR soilsuite | |
| 25.07.2025| Processed MeteoSwiss 2024 variables to cubes | |
| 25.07.2025| Downloaded PlanetLabs data for some AOIs | | 
| 28.07.2025| Downloaded S2 until 2025-07 for some regions | Package versions: minicuber ([commit version](https://github.com/EOA-team/minicuber/tree/f201746395ed9e088ba9ef806e5e9f5c87ad2460)) with local sen2nbar provided in minicuber repo| 
| 17.10.2025| Processed ccsols data | | 
| 11.11.2025| Processed snowdepth data | | 
| 07.04.2026| Downloaded S2 for 2025 | Package versions: sen2nbar==2023.8.1  minicuber ([commit version](https://github.com/EOA-team/minicuber/tree/686227f8fc0131053a448a3db59a6054e44c08da))| 

---


## Storage structure <a name="storage-structure"></a>

```
📁 \\eo-nas1\data
├── satellite
│   ├── sentinel2
│   │   ├── raw
│   │   │   └── CH                    # raw S2 zarr cubes
│   │   ├── coreg
│   │   │   └── CH                    # coregistered S2 zarr cubes
│   │   └── DLR_soilsuite             # DLR bare-soil composite
│   ├── PlanetLabs
│   │   └── raw
│   │       └── <AOI_name>
│   └── landsat
│       └── raw
│           └── CH
│               ├── 45                # Landsat 4-5
│               └── 89                # Landsat 8-9
│
├── swisstopo
│   ├── SwissImage
│   │   ├── raw
│   │   │   ├── 10cm
│   │   │   └── 2m
│   │   └── cubes
│   │       ├── 10cm
│   │       └── 2m
│   └── DEM                           # swissalti3D zarr cubes
│
├── meteo
│   ├── Rhires
│   ├── Srel
│   ├── Tabs
│   ├── Tmax
│   ├── Tmin
│   └── snowdepth
│
└── soil
    └── ccsols
        ├── Daten_2024-01             # raw rasters
        └── datacubes                 # processed zarr cubes
```

---
### Tools

Code examples and function to extract the data and process the zarr files with the `xarray` package can be found [here](https://github.com/EOA-team/SwissEODI_utils)