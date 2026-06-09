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
  tables.py        # table_s1(), tables_s2_s9()
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

(`environment.yml` was exported from the working `hm_plots` env. `pytest` is not
included; the test files are plain assertion functions and can also be run with
`pytest` if you install it.)

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
- **`USE_DASK`** (default `True`) — start a Dask client for the heavier rasters.

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
| Table S1 | `table_s1` | `Supplementary_Table_S1_covariates.pdf` |
| Tables S2–S9 | `tables_s2_s9` | `realm_tables/*.csv`, `realm_tables.pdf` |

## Pipeline dependency

`fig10` computes `unprotected_loss_stats.csv`; `tables_s2_s9` consumes it (and runs
`fig10` first automatically if the CSV is missing). `make_paper.py` orders Figure 10
before Tables S2–S9.

## Data availability

Inputs are read from `config.DATA_DIR` (`../data`). At the time this repo was built,
these inputs were **present**, so their outputs run as-is:

- Figs 2, 3 · Fig S1 · Figs 8, 9 · Figs S5, S6 · Fig 10 · Table S1 · Tables S2–S9

The following inputs were **pruned** from `data/` (only `.aux.xml` sidecars remained):
the `*_AA_1000.tiff` observed series and the `prediction_*_blended.tif` predictions.
Until they are restored, these outputs are code-complete but will not run:

- Figs S2, S3, S4 · Figs 4, 5 · Fig SX · Fig 6 · Fig 7

## Network

`figSX`, `fig6`, and `fig7` fetch satellite basemaps at runtime from the ESRI World
Imagery REST API, and use cartopy's `stock_img()`/`coastlines()` (Natural Earth data
is downloaded on first use). An internet connection is required for those three.

## Provenance

Figure bodies were ported from the standalone `plot_*.py` scripts (themselves
extracted from the original plotting notebooks). Shared scaffolding was factored into
`utils.py` and parameters into `config.py` without changing visual output: Figs 2, 3,
S1, 8, 9, S5, and S6 were verified pixel-identical against the originals where input
data was available.
