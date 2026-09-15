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
                   #   ternary colour scheme, fan legend, coordinate formatting
  figures.py       # one function per figure group (fig2_3 … fig10, figS1_S4, figSX, figS5, figS6)
  tables.py        # table_1(), tables_s1_s8()
  stats.py         # standalone console summary statistics (not part of make_paper.py)
  equal_area.py    # shared equal-area analysis grid (leaf module: numpy + rasterio only)
  quantiles.py     # per-pixel quantile functions from the icechunk stores (leaf module)
  tests/           # unit tests for the pure helpers, the stats accumulators and config
  environment.yml  # exported hm_plots conda/mamba environment
  output/          # generated figures/tables (gitignored)
```

## What the model produces

The forecasts come from a distributional ConvLSTM. It does not emit a point
prediction; it emits a **monotone spline quantile function per pixel**,
published at 64 levels. Three kinds of input are derived from it and all three
live under `data/conv_dist/`:

| Input | What it is |
|---|---|
| `forecast/forecast_qf.icechunk` | the full quantile functions, 2025–2040 (base year 2020) |
| `hindcast/hindcast_qf.icechunk` | the full quantile functions, 2005–2020 (base year 2000) |
| `cogs/HM_p10_YYYY.tif` | P(HM ≥ 0.10) per pixel |
| `cogs/HM_p40_YYYY.tif` | P(HM ≥ 0.40) per pixel |

The blended `*_central_blended.tif` rasters are **E[Q] = ∫Q(u)du — the mean**,
not the median, so `stats.py` uses them directly as the E[HM] layer, Fig 2
labels its axis "expected" and Fig 4's third bar is labelled "Predicted (mean)".

Two traps are worth knowing about before touching this data:

- **The icechunk stores do not self-decode.** `quantile_forecast` is `int16`
  with attributes named `scale` and `sentinel`, which are *not* the CF names
  (`scale_factor`, `_FillValue`) xarray decodes automatically. Read them through
  `quantiles.decode()`; a raw read is ~32,000× too large and passes every
  threshold test silently.
- **The probability COGs need coordinate snapping.** They declare the same
  transform and shape as the HM rasters but were written by a different tool, so
  their coordinate arrays differ in the last bits (~1e-11°). xarray aligns on
  equality, so combining them natively inner-joins 40,000 longitudes down to 42
  and returns a near-empty array with no warning. Use `utils.align_like()` on
  the native grid. Anything read through `equal_area.open_equal_area` is warped
  onto one explicitly-built grid and is immune.

## Expected areas, and why the scenario columns are gone

Every area this repo reports is now an **expected area**: the sum of per-pixel
probabilities times cell area. That is valid under any spatial dependence
between pixels, because expectation is linear.

The previous version reported a "lower" and an "upper" figure alongside a
central one. Those were the 2.5th and 97.5th percentiles of each pixel's own
distribution, thresholded and counted — so the "upper" number was the area lost
only if every pixel realised its unlucky outcome *simultaneously*. That is the
perfect-dependence extreme, and the largest value obtainable under any
dependence structure, not an upper bound on a forecast.

`stats.py` still brackets each expected area, but with two clearly-named ends:

- **independence** — `1.96 × √Σp(1−p)`, the narrowest interval that is plausible
- **perfect dependence** — the area with `p > 0.975` and with `p > 0.025`, which
  are exactly the old "lower" and "upper" numbers

The truth sits between them, and both ends need only the marginal
probabilities.

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
- **`OUTPUT_DIR`** — where outputs are written (default: `./output_new`).
- **`FIG7_SEEDS`** (default `[0]`) and **`FIGSX_N_PLOTS`** (default `1`) — these figures
  emit one image per seed / per random site-pair. The notebooks produced 100 and 10
  respectively; bump these to reproduce more, or pin a specific seed.
- **`FIG5_N_BLOCKS`** (default `48`) — how many chunk-aligned 512×512 tiles Figures 4
  and 5 draw from the hindcast quantile store. Each is exactly one icechunk chunk.
- **`FIG5_PIT_BINS`** (default `7`) — bins in Figure 5's PIT histogram.
- **`FIG7_DENSITY_XLIM`** (default `(-0.03, 0.5)`) — x-limits shared by Figure 7's
  predictive-density row.
- **`VALIDATION_SPLIT`** (default `None`) — the previous model used a spatial split and
  `split_mask == 2` was the held-out set. The k-fold model stitches holdout predictions,
  so every pixel is out of sample and Figs 4/5 use all land. Set to `2` for the old sample.
- **`LOW_CUT`** / **`HIGH_CUT`** (0.10 / 0.40), **`P_LOSS_BREAKS`** (0.025 / 0.5) and
  **`PROTECTION_TARGET`** (30%) — the condition-class cut-points and the Figure 10 brackets.
  The cuts are closed at the bottom: natural is `HM <= LOW_CUT`, moderately modified
  is `(LOW_CUT, HIGH_CUT)`.
- **`P_EXCEED_LEVELS`** / **`P_EXCEED_COLORS`** — the uneven-break probability scale shared
  by Figures 3 and 6. A linear scale puts nearly every land pixel in one colour.
- **`DPI_MAP`** (600) / **`DPI_PLOT`** (300), colours, clim, inset locations and
  ternary thresholds.
- **`INSET_MAP_FRAC`**, **`INSET_REF_MAP_W_IN`** and the four `INSET_*_LW` /
  `INSET_MARKER_*` sizes — inset geometry, all measured against the **map axes'
  width**, not the figure. Figure 2 comes out of `hv.render()` on a 4×4 in canvas
  while Figures 3a/3b build their own 13.9×7.6 in one, so a figure-fraction size
  and a point-valued stroke both render ~3.2× differently between them. Scaling
  against Figure 2's 2.8 in map reproduces Figure 2 exactly and makes every other
  map match it.
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
| Figures 3a, 3b | `fig2_3` | `fig3a_p10_map.png`, `fig3b_p40_map.png` |
| Figures S1–S4 | `figS1_S4` | `figS1_hmdiff_map.png`, `figS2_hm2025_map.png`, `figS3_hm2030_map.png`, `figS4_hm2035_map.png` |
| Figures 4, 5 | `fig4_5` | `fig4_obs_vs_pred_dist.png`, `fig5_pit_calibration.png` |
| Figure SX | `figSX` | `fig_SX_obs_vs_predicted_random_NN.png` (×`FIGSX_N_PLOTS`) |
| Figure 6 | `fig6` | `fig_6_obs_vs_predicted_predetermined.png` |
| Figure 7 | `fig7` | `fig_7_timeseries_site2seedNN.png` (one per `FIG7_SEEDS`), `fig7_fan_legend.png` |
| Figure 8 | `fig8_9` | `fig8_hexbin_matrix_lower.png` |
| Figure 9 | `fig8_9` | `fig9_rgb_map.png`, `fig9_ternary_alpha_legend.png` |
| Figure S5 | `figS5` | `figS5_hmdiff_lower_map.png` |
| Figure S6 | `figS6` | `figS6_hmdiff_upper_map.png` |
| Figure 10 | `fig10` | `fig_unprotected_loss_map.png`, `fig_unprotected_loss_radial.png`, `unprotected_loss_stats.csv`, `realm_loss_stats.csv` |
| Table 1 | `table_1` | `Table_1_covariates.pdf` |
| Tables S1–S8 | `tables_s1_s8` | `realm_tables/*.csv`, `realm_tables.pdf` |

Figure 3 was a five-class categorical map built from the lower/central/upper
triple; it is now two exceedance-probability maps, and `data/raster_classes.tif`
is no longer read by anything. Figure 10 ran its whole analysis twice, once per
scenario, and now runs once with unprotected natural land split by its
probability of loss.

Figure 4 was a hexbin of observed against expected change; it is now the change
histogram, with three bars per bin - observed, predicted from the full
distribution, and predicted from the mean surface. The PIT calibration
histogram that was panel (b) of that figure is now Figure 5 on its own, at
`FIG5_PIT_BINS` bins. The hexbin is no longer rendered.

Figures S5 and S6 were 11 and 22 regional zooms. They are now single global
maps in the Figure 2 mould - same ramp, same `CLIM`, same three circular insets
- showing the lower (2.5th percentile) and upper (97.5th) ends of each pixel's
predictive distribution. Read them as a triptych with Figure 2's mean in the
middle; neither bound is a scenario. `config.REGIONS` is no longer read.

The 2020 condition class below the first cut is called **natural** throughout,
and it is `HM <= 0.10` — closed at the cut, so a pixel sitting exactly on 0.10
is natural. Moderately modified is `(0.10, 0.40)`. Every base mask in the repo
uses that convention, so the three classes still partition the land.

The word "intact" is gone with it, including from the CSV schemas.

## Areas, and the CSV schema

Every area `fig10` reports is **km², and says so in its column name**. The
zonal sums are counts of equal-area cells, multiplied through by the cell
area once, at the point the DataFrame is built — so nothing downstream has to
remember a unit. Derived shares keep the bare stem under a `pct_` prefix,
since a percentage has no unit, and they are unchanged by the scaling because
the cell area cancels out of a ratio.

Both CSVs therefore changed schema. `unprotected_loss_stats.csv`:
`eco_land_total` → `eco_land_km2` (the old duplicate `eco_land_km2` is gone),
`protected_land` → `protected_km2`, `intact_total` → `natural_km2`,
`unprot_nonnatural_2020` → `…_km2`, `intact_unprot_p_*` →
`natural_unprot_p_*_km2`. `realm_loss_stats.csv`: `intact_lost_YYYY_ha` →
`natural_lost_YYYY_km2` and `midmod_lost_YYYY_ha` → `…_km2`, so Tables S1–S8
are km² throughout instead of hectares beside a km² total.

So **an `output/` left over from an earlier run must be regenerated**:
`tables_s1_s8` stops with a "missing required columns" error against an old
`realm_loss_stats.csv` rather than silently mixing schemas. The numbers change
too, because the cut moved.

## Pipeline dependency

`fig10` computes `unprotected_loss_stats.csv` (protection target vs loss risk) and
`realm_loss_stats.csv` (expected loss by year); `tables_s1_s8` consumes the latter
(and runs `fig10` first automatically if it is missing). `make_paper.py` orders
Figure 10 before Tables S1–S8.


## Network

`figSX`, `fig6`, and `fig7` fetch satellite basemaps at runtime from the ESRI World
Imagery REST API, and use cartopy's `stock_img()`/`coastlines()` (Natural Earth data
is downloaded on first use). An internet connection is required for those three.

