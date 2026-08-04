# hm_forecasting_paper

Reproducible figures and tables for the human-modification (HM) forecasts paper.
A single command regenerates every figure and supplementary table; a config file
chooses which outputs to build.

## Repo layout

```
hm_forecasting_paper/
  make_paper.py    # entry-point: runs the outputs enabled in config.RUN, in dependency order
  config.py        # all paths, parameters, colours, multi-image counts, and the RUN toggles
  utils.py         # shared helpers: colormaps, global-map + inset builders, ESRI fetch,
                   #   ternary colour scheme, coordinate formatting
  figures.py       # one function per figure group (fig2_3 … fig10, figS1_S4, figSX, figS5, figS6)
  tables.py        # table_1(), tables_s1_s8()
  stats.py         # standalone console summary statistics (not part of make_paper.py)
  equal_area.py    # shared equal-area analysis grid (leaf module: numpy + rasterio only)
  tests/           # lightweight unit tests for the pure utils + config + orchestrator
  environment.yml  # exported hm_plots conda/mamba environment
  output/          # generated figures/tables (gitignored)
```

## Environment

The code runs in the `hm_plots` mamba/conda environment:

```bash
mamba env create -f environment.yml   # or: conda env create -f environment.yml
mamba activate hm_plots
```


## Configure

Everything tunable lives in `config.py`:

- **`RUN`** — a dict of booleans, one per output. Set any to `False` to skip it.
- **`DATA_DIR`** / **`PATHS`** — input locations (default: the sibling `../data` folder).
- **`OUTPUT_DIR`** — where outputs are written (default: `./output`).
- **`FIG7_SEEDS`** (default `[0]`) and **`FIGSX_N_PLOTS`** (default `1`) — these figures
  emit one image per seed / per random site-pair. The notebooks produced 100 and 10
  respectively; bump these to reproduce more, or pin a specific seed.
- **`DPI_MAP`** (600) / **`DPI_PLOT`** (300), colours, clim, inset locations, ternary
  thresholds, and the S5/S6 region list.
- **`STOP_ON_ERROR`** (default `False`) — keep going if one output fails.
- **`USE_DASK`** (default `False`) — start a Dask client for the heavier rasters.
- **`TABLE_FORMAT`** (default `"pdf"`) — `"pdf"`, `"docx"`, or `"both"` for Table 1 and Tables S1–S8.
- **`EQUAL_AREA_CRS`** / **`EQUAL_AREA_RES`** / **`EQUAL_AREA_RESAMPLING`**
  (default `"EPSG:6933"` / `1000` m / `"nearest"`) — see **Area weighting** below.
  Set `EQUAL_AREA_CRS = None` to revert to native-grid pixel counts.

## Area weighting

The source rasters are EPSG:4326 at 0.009°. Every reported quantity is computed after warping the inputs onto a shared equal-area grid (EPSG:6933, NSIDC EASE-Grid 2.0 Global, at 1 km, nearest-neighbour).


## Run

```bash
mamba run -n hm_plots python make_paper.py
```

Outputs are written to `output/`. The run logs each output's start/finish/timing
and prints a summary of what ran, was skipped, or failed.

## Outputs

| Output | Function | Files (in `output/`) |
|---|---|---|
| Figure 2 | `fig2_3` | `fig2_hmdiff_map.png` |
| Figure 3 | `fig2_3` | `fig3_uncer_map_hr.png` |
| Figures S1–S4 | `figS1_S4` | `figS1_hmdiff_map.png`, `figS2_hm2025_map.png`, `figS3_hm2030_map.png`, `figS4_hm2035_map.png` |
| Figures 4, 5 | `fig4_5` | `fig4_obs_vs_pred_change_hexbin.png`, `fig5_obs_vs_pred_hist.png` |
| Figure SX | `figSX` | `fig_SX_obs_vs_predicted_random_NN.png` (×`FIGSX_N_PLOTS`) |
| Figure 6 | `fig6` | `fig_6_obs_vs_predicted_predetermined` |
| Figure 7 | `fig7` | `fig_7_timeseries_site2seedNN.png` (one per `FIG7_SEEDS`) |
| Figure 8 | `fig8_9` | `fig8_hexbin_matrix_lower.png` |
| Figure 9 | `fig8_9` | `fig9_rgb_map.png`, `fig9_ternary_alpha_legend.png` |
| Figure S5 | `figS5` | `figS5_<region>.png` (×11) |
| Figure S6 | `figS6` | `figS6_<region>.png` (×11) |
| Figure 10 | `fig10` | `fig_unprotected_loss_map_{central,upper}.png`, `fig_unprotected_loss_radial_{central,upper}.png`, `unprotected_loss_stats.csv` |
| Table 1 | `table_1` | `Table_1_covariates.pdf` |
| Tables S1–S8 | `tables_s1_s8` | `realm_tables/*.csv`, `realm_tables.pdf` |

## Pipeline dependency

`fig10` computes `unprotected_loss_stats.csv`; `tables_s1_s8` consumes it (and runs
`fig10` first automatically if the CSV is missing). `make_paper.py` orders Figure 10
before Tables S1–S8.


## Network

`figSX`, `fig6`, and `fig7` fetch satellite basemaps at runtime from the ESRI World
Imagery REST API, and use cartopy's `stock_img()`/`coastlines()` (Natural Earth data
is downloaded on first use). An internet connection is required for those three.

