"""Figure-generating functions. Each writes its outputs to config.OUTPUT_DIR.

Bodies are ported from the standalone plot_*.py scripts; the shared global-map /
inset scaffolding (Figs 2, S1-S4) is delegated to utils, while Fig 3's
categorical map and Figs 8/9's RGB map keep their own verbatim bodies.

Area weighting: the source rasters are EPSG:4326, on which a cell's ground area
falls off with cos(lat), so a pixel tally is not an area. Every *number* here -
the per-ecoregion shares in fig10, the densities in Figs 4/5, and the sample
behind Fig 8's panels, the Spearman matrix and Fig 9's colour cut-points - is
computed after warping onto config.EQUAL_AREA_CRS, where one cell is one
constant patch of ground. Maps are deliberately left on the native grid, since
cartopy already renders them area-honestly.
"""
import contextlib
from pathlib import Path
import numpy as np
import xarray as xr
import rioxarray as rxr
import rasterio
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib.path as mpath
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap
import hvplot.xarray  # noqa: F401  (registers .hvplot accessor)
import holoviews as hv

import config
import quantiles
import utils
from utils import (fetch_esri_satellite, fmt_coord,
                   transform_xarray_layer, create_ternary_alpha_array)

OUT = config.OUTPUT_DIR


def _load_exceedance(year, level, *, base_year=2020, chunks='auto'):
    """P(HM >= level) for `year`, masked to the pixels the base admits.

    Returns (probability, base, land) as native-grid DataArrays. `base` is the
    condition class in `base_year` that the probability is about - natural for
    the 0.10 level, moderately modified for 0.40 - and `land` is everything
    with an observed value. Keeping the three separate is what lets the maps
    show ocean, out-of-base land and probability as three distinct states
    rather than folding the middle one into the first.

    Both cuts are closed at the bottom (natural is HM <= 0.10), so the two
    bases stay disjoint and together with "already >= 0.40" they partition the
    land. See config.LOW_CUT.
    """
    key = 'hm_p10' if level == config.LOW_CUT else 'hm_p40'
    p = rxr.open_rasterio(config.PATHS[f'{key}_{year}'], chunks=chunks,
                          masked=True).squeeze(drop=True)
    hm = rxr.open_rasterio(config.PATHS[f'hm_{base_year}_aa'], chunks=chunks,
                           masked=True).squeeze(drop=True)
    # Same grid, coordinates built by a different tool - see utils.align_like.
    # Without this the masking below silently inner-joins to a few hundred cells.
    p = utils.align_like(p, hm)
    land = hm.notnull()
    if level == config.LOW_CUT:
        base = land & (hm <= config.LOW_CUT)
    else:
        base = land & (hm > config.LOW_CUT) & (hm < config.HIGH_CUT)
    return p.where(base), base, land


def _exceedance_display(p_base, land, stride):
    """Coarsen a probability field and its land mask for display.

    The probability is averaged, not sub-sampled. At this stride one displayed
    cell covers stride^2 source cells, and the mean probability over a block is
    the expected fraction of it that crosses the threshold - the same quantity
    the rest of the paper reports. Taking the maximum would show the worst
    pixel in every block and inflate the map; striding would drop isolated hot
    pixels entirely.
    """
    kw = dict(x=stride, y=stride, boundary='trim')
    disp = p_base.coarsen(**kw).mean(skipna=True).compute()
    land_c = land.coarsen(**kw).max().compute()
    return disp, land_c


def _exceedance_colorbar(fig, ax, mesh, label, *, shrink=0.42, pad=0.02,
                         orientation='vertical', fontsize=9):
    """Discrete colorbar for the exceedance scale, ticked on the breaks."""
    cbar = fig.colorbar(mesh, ax=ax, orientation=orientation, shrink=shrink,
                        pad=pad, ticks=config.P_EXCEED_LEVELS,
                        spacing='uniform')
    cbar.set_label(label, fontsize=fontsize)
    labels = [f'{t:g}' for t in config.P_EXCEED_LEVELS]
    if orientation == 'vertical':
        cbar.ax.set_yticklabels(labels)
    else:
        cbar.ax.set_xticklabels(labels)
    cbar.ax.tick_params(labelsize=max(6, fontsize - 2), width=0.3, length=2)
    cbar.outline.set_linewidth(0.3)
    return cbar


def _exceedance_map(disp, land_c, *, cbar_label, out_path,
                    p_full=None, land_full=None):
    """Global Robinson map of an exceedance probability, with the Fig 2 insets."""
    cmap, norm = utils.exceedance_cmap_norm()

    fig = plt.figure(figsize=(13.9, 7.6))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.set_facecolor(config.OCEAN_HEX)

    # Land first, as one flat colour, so a NaN in the probability layer reads
    # as "already above the threshold in the base year", not as ocean.
    ax.pcolormesh(
        land_c.x.values, land_c.y.values,
        np.where(np.asarray(land_c.values), 1.0, np.nan),
        cmap=mcolors.ListedColormap([config.OUT_OF_BASE_HEX]),
        vmin=0.0, vmax=1.0, transform=ccrs.PlateCarree(),
        shading='auto', rasterized=True, zorder=1,
    )
    mesh = ax.pcolormesh(
        disp.x.values, disp.y.values, disp.values,
        cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
        shading='auto', rasterized=True, zorder=2,
    )

    ax.spines['geo'].set_visible(True)
    ax.spines['geo'].set_edgecolor('black')
    ax.spines['geo'].set_linewidth(0.3)
    ax.set_title('')

    _exceedance_colorbar(fig, ax, mesh, cbar_label)

    if p_full is not None:
        utils.add_circular_insets(fig, ax, p_full, cmap=cmap, norm=norm,
                                  under=land_full)

    fig.savefig(out_path, dpi=config.DPI_MAP, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path.name}')


def fig2_3():
    """Fig 2 (expected HM change 2020-2040) and Figs 3a/3b (exceedance maps).

    Figure 3 used to be a five-class categorical map built from the lower /
    central / upper triple, which encoded a scenario rather than a probability.
    It is now two maps of what the model actually estimates: for land natural in
    2020, the probability it reaches HM >= 0.10 by 2040; and for land that is
    moderately modified, the probability it reaches HM >= 0.40.
    data/raster_classes.tif is no longer read by anything.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    utils.register_coolwarm_cmap()

    # ---- Figure 2 (shared global-map builder) ----
    # The "central" surface is E[Q] = the mean of the predictive distribution,
    # so this difference is an expected change, not a median scenario.
    ds = utils.load_hm_diff()
    ds = ds.to_dataset(name="hm")
    ds = ds.drop('band')
    ds = ds.squeeze()
    mask = ((ds['hm'] >= -1) & (ds['hm'] <= 1)).compute()
    ds = ds.where(mask, drop=True)
    ds['hm'] = ds['hm'].chunk({'x': 1024, 'y': 1024})
    pds = ds['hm'].compute()

    fig, ax = utils.build_global_robinson_map(
        pds, cmap='my_custom_coolwarm', clim=config.CLIM,
        cbar_label='Expected HM change 2020-2040')
    utils.add_circular_insets(fig, ax, pds, cmap='my_custom_coolwarm',
                              vmin=config.CLIM[0], vmax=config.CLIM[1])
    fig.savefig(OUT / 'fig2_hmdiff_map.png', dpi=config.DPI_MAP, bbox_inches='tight')
    plt.close(fig)
    del ds, pds

    # ---- Figures 3a / 3b: exceedance probabilities in 2040 ----
    year = config.FORECAST_YEARS[-1]
    stride = 8                     # same decimation the categorical map used

    # The base each probability is conditioned on (natural / moderately
    # modified in 2020) is carried by the map itself - out-of-base land is drawn
    # in its own flat colour - and by the caption, so the colorbar states only
    # the event. The strict inequality is exact, not a loosening: the predictive
    # distribution is a continuous spline, so P(HM >= c) and P(HM > c) are the
    # same number, and "> 0.1" is the one that matches "natural is HM <= 0.1".
    for panel, level, label in (
        ('a', config.LOW_CUT, f'P(HM > {config.LOW_CUT:g})'),
        ('b', config.HIGH_CUT, f'P(HM > {config.HIGH_CUT:g})'),
    ):
        p_base, base, land = _load_exceedance(year, level)
        disp, land_c = _exceedance_display(p_base, land, stride)
        _exceedance_map(
            disp, land_c, cbar_label=label,
            out_path=OUT / f'fig3{panel}_p{int(level * 100):02d}_map.png',
            p_full=p_base, land_full=land,
        )
        del p_base, base, land, disp, land_c



def figS1_S4():
    """Fig S1 (observed 2020-2000 diff) + Figs S2/S3/S4 (2025/2030/2035 - 2020).

    S2-S4 were three byte-identical notebook blocks differing only by prediction
    year, colorbar label, and output name; reproduced here as one loop driven by
    config.FIGS_YEAR_VARIANTS.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    utils.register_coolwarm_cmap()

    # ---- Figure S1: observed HM change 2020-2000 ----
    dsobs = utils.load_hm_diff_obs()
    dsobs = dsobs.to_dataset(name="hm")
    ds = dsobs.drop('band')
    ds = ds.squeeze()
    mask = ((ds['hm'] >= -1) & (ds['hm'] <= 1)).compute()
    ds = ds.where(mask, drop=True)
    ds['hm'] = ds['hm'].chunk({'x': 1024, 'y': 1024})
    pds = ds['hm'].compute()

    fig, ax = utils.build_global_robinson_map(
        pds, cmap='my_custom_coolwarm', clim=config.CLIM,
        cbar_label='HM change 2000-2020', hide_geo_spine=True)
    utils.add_circular_insets(fig, ax, pds, cmap='my_custom_coolwarm',
                              vmin=config.CLIM[0], vmax=config.CLIM[1])
    fig.savefig(OUT / 'figS1_hmdiff_map.png', dpi=config.DPI_MAP, bbox_inches='tight')

    # ---- Figures S2/S3/S4: predicted HM change YEAR-2020 ----
    ds2020 = rxr.open_rasterio(config.PATHS['hm_2020_aa'], chunks='auto')
    for pred_key, label, fname in config.FIGS_YEAR_VARIANTS:
        pred = rxr.open_rasterio(config.PATHS[pred_key], chunks='auto')
        ds = pred - ds2020
        ds = ds.to_dataset(name="hm")
        ds = ds.drop('band')
        ds = ds.squeeze()
        mask = ((ds['hm'] >= -1) & (ds['hm'] <= 1)).compute()
        ds = ds.where(mask, drop=True)
        ds['hm'] = ds['hm'].chunk({'x': 1024, 'y': 1024})
        pds = ds['hm'].compute()

        fig, ax = utils.build_global_robinson_map(
            pds, cmap='my_custom_coolwarm', clim=config.CLIM, cbar_label=label)
        utils.add_circular_insets(fig, ax, pds, cmap='my_custom_coolwarm',
                                  vmin=config.CLIM[0], vmax=config.CLIM[1])
        fig.savefig(OUT / fname, dpi=config.DPI_MAP, bbox_inches='tight')


# --- Figures 4 and 5: predicted distributions against the observations -------
#
# Both are built from one pass over the hindcast quantile store (_fig4_sample).
# Figure 4 is the change histogram - observed, predicted-from-the-distribution
# and predicted-from-the-mean - and Figure 5 is the PIT calibration histogram,
# which used to be panel (b) of the same figure and now stands alone.
#
# The hexbin of observed against expected change that used to be Figure 4 is no
# longer rendered.

# The bins Figure 4 has always used, and the labels that go with them.
FIG4_BIN_EDGES = np.array([-1.0, -0.05, 0.0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
FIG4_BIN_LABELS = [
    '-1 to -0.05', '-0.05 to 0', '0 to 0.005', '0.005 to 0.02',
    '0.02 to 0.05', '0.05 to 0.1', '0.1 to 0.2', '0.2 to 0.5', '0.5 to 1',
]

# Bar order, colour and legend text. The two predicted bars are deliberately
# shades of one hue: they are the same forecast read two ways, and the gap
# between them is the figure's point - the mean is a single number per pixel
# and lands in one bin, while the distribution spreads that pixel's weight
# across every bin it reaches.
FIG4_SERIES = [
    ('observed',      'Observed',                '#58c785'),
    ('expected',      'Predicted (distribution)', '#5fa0d3'),
    ('central',       'Predicted (mean)',        '#22405c'),
]


def _row_area_km2(lats, dlat=0.009, dlon=0.009):
    """Ground area of one native cell in each row, km2.

    A geographic cell spanning [lat1, lat2] x dlon has exact spherical area
    R^2 * dlon * (sin lat2 - sin lat1). Using it as a per-pixel weight makes
    the sample area-fair without warping anything, which matters because the
    quantile store is only readable on its native grid.
    """
    r = utils.AUTHALIC_RADIUS_M
    top = np.radians(lats + dlat / 2.0)
    bottom = np.radians(lats - dlat / 2.0)
    return (r ** 2 * np.radians(dlon) * (np.sin(top) - np.sin(bottom))) / 1e6


def _fig4_sample():
    """Accumulate Figures 4 and 5 from the hindcast quantile store.

    Reads chunk-aligned 512x512 tiles - each is exactly one icechunk chunk, so
    a tile costs one read rather than four - and folds each straight into the
    accumulators. The raw quantile functions are never all held at once: a
    million pixels x 64 levels would be gigabytes, and nothing downstream needs
    them after the per-tile reduction.

    Returns a dict with, per bin, the observed weighted count, the expected
    weighted count under the full distributions and the weighted count under
    the central (mean) surface; plus the PIT values and their weights.

    The three bar series come off the *same* tiles and the same weights, which
    is the only way the comparison is honest: the central surface is read from
    its own raster in the same window rather than from a separately-sampled
    equal-area pass, so nothing but the estimator differs between the bars.

    Weights are the ground area of each native cell in equal-area cell
    equivalents, so a bar height keeps meaning "1 km equal-area cells" as it
    always has.
    """
    ds = quantiles.open_qf('hindcast')
    p = quantiles.levels(ds)
    times = np.asarray(ds['time'].values)
    t_index = int(np.argmin(np.abs(times - 2020)))
    store_lat = np.asarray(ds['latitude'].values)

    size = quantiles.CHUNK
    cell_km2 = (config.EQUAL_AREA_RES / 1000.0) ** 2
    rng = np.random.default_rng(config.FIG5_SEED)

    n_bins = len(FIG4_BIN_EDGES) - 1
    acc = {
        'observed': np.zeros(n_bins), 'expected': np.zeros(n_bins),
        'central': np.zeros(n_bins), 'pit': [], 'pit_w': [],
        'n_pixels': 0, 'n_blocks': 0, 'area_km2': 0.0,
    }

    with contextlib.ExitStack() as stack:
        hm00 = stack.enter_context(rasterio.open(config.PATHS['hm_2000_aa']))
        hm20 = stack.enter_context(rasterio.open(config.PATHS['hm_2020_aa']))
        # E[Q] for 2020, the mean of the same predictive distribution the
        # quantile store holds - so the third bar is a different summary of one
        # forecast, not a different forecast.
        cen20 = stack.enter_context(
            rasterio.open(config.PATHS['pred_2020_central']))
        split = (stack.enter_context(rasterio.open(config.PATHS['split_mask']))
                 if config.VALIDATION_SPLIT is not None else None)

        n_rows, n_cols = hm20.height // size, hm20.width // size
        order = rng.permutation(n_rows * n_cols)

        for flat in order:
            if acc['n_blocks'] >= config.FIG5_N_BLOCKS:
                break
            r0 = int(flat // n_cols) * size
            c0 = int(flat % n_cols) * size
            window = rasterio.windows.Window(c0, r0, size, size)

            # Cheap rasters first: skip a tile before paying for its 33 MB
            # quantile chunk.
            if split is None:
                keep = np.ones((size, size), dtype=bool)
            else:
                keep = split.read(1, window=window) == config.VALIDATION_SPLIT
                if keep.sum() < 1000:
                    continue
            y0 = utils.read_masked(hm00, window)
            y1 = utils.read_masked(hm20, window)
            yc = utils.read_masked(cen20, window)
            keep &= np.isfinite(y0) & np.isfinite(y1) & np.isfinite(yc)
            if keep.sum() < 1000:
                continue

            v = quantiles.block_qf(ds, r0, c0, t_index, size=size)
            keep &= np.isfinite(v).all(axis=0)
            n = int(keep.sum())
            if n < 1000:
                continue

            rows = np.nonzero(keep)[0]
            w = _row_area_km2(store_lat[r0:r0 + size])[rows] / cell_km2
            vk = v[:, keep].astype(np.float64)
            base = y0[keep].astype(np.float64)
            obs_change = y1[keep].astype(np.float64) - base
            cen_change = yc[keep].astype(np.float64) - base

            # The predictive distribution is over the HM *level* in 2020, and
            # the 2000 observation is a constant per pixel, so the distribution
            # of change is the same one with its axis shifted. Shifting the bin
            # edges per pixel is exactly equivalent and costs nothing.
            probs = quantiles.bin_probs(p, vk, FIG4_BIN_EDGES[:, None] + base[None, :])

            acc['expected'] += (probs * w).sum(axis=1)
            acc['observed'] += np.histogram(obs_change, bins=FIG4_BIN_EDGES,
                                            weights=w)[0]
            # The mean surface collapses each pixel's distribution to one
            # number, so this is an ordinary histogram of a point prediction -
            # the same tally the observed bar gets.
            acc['central'] += np.histogram(cen_change, bins=FIG4_BIN_EDGES,
                                           weights=w)[0]
            # PIT of the observed 2020 level under its own predictive law -
            # identical to the PIT of the change, since the shift cancels.
            acc['pit'].append(quantiles.pit(p, vk, y1[keep].astype(np.float64), rng))
            acc['pit_w'].append(w)
            acc['n_pixels'] += n
            acc['area_km2'] += float(w.sum()) * cell_km2
            acc['n_blocks'] += 1
            del v, vk, probs

    acc['pit'] = np.concatenate(acc['pit']) if acc['pit'] else np.array([])
    acc['pit_w'] = np.concatenate(acc['pit_w']) if acc['pit_w'] else np.array([])
    scope = ('all land (k-fold holdout, every pixel out of sample)'
             if config.VALIDATION_SPLIT is None
             else f'split_mask == {config.VALIDATION_SPLIT}')
    print(f"  Fig 4/5 sample: {acc['n_pixels']:,} pixels from "
          f"{acc['n_blocks']} tiles ({acc['area_km2']:,.0f} km2); {scope}")
    return acc


def _fig4_plot(acc, out_path):
    """Figure 4: the observed change histogram against two readings of the forecast.

    Three bars per bin, all tallies of the same cells with the same area
    weights:

      Observed                 where the 2020 observation actually landed
      Predicted (distribution) the expected count - the sum over pixels of that
                               pixel's probability of landing in the bin, so one
                               pixel contributes a fraction to several bins
      Predicted (mean)         the same forecast collapsed to E[Q] per pixel and
                               tallied like an observation, so one pixel
                               contributes 1 to exactly one bin

    The third bar is what the old point-prediction figure showed, and it sits
    here to make the cost of collapsing a distribution visible: a mean
    concentrates mass near the middle of the range and under-fills both tails,
    while the distribution bar reproduces them.

    The axis stays logarithmic - the bins span five orders of magnitude - but
    the label no longer says so, because the tick labels already do.
    """
    x = np.arange(len(FIG4_BIN_LABELS))
    width = 0.8 / len(FIG4_SERIES)
    offsets = (np.arange(len(FIG4_SERIES)) - (len(FIG4_SERIES) - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(10.5, 5.2))

    for (key, label, colour), off in zip(FIG4_SERIES, offsets):
        ax.bar(x + off, acc[key], width, color=colour, label=label)

    ax.set_yscale('log')
    ax.set_xlabel('HM change (2000 - 2020)', fontweight='bold', fontsize=13)
    ax.set_ylabel('Pixel count', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(FIG4_BIN_LABELS, rotation=45, ha='right', fontsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=config.DPI_PLOT, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path.name}')


def _fig5_plot(acc, out_path, n_bins=None):
    """Figure 5: PIT calibration, previously panel (b) of Figure 4.

    The PIT is uniform exactly when each pixel's own predictive distribution is
    right, which is the thing the bar chart cannot show - a set of forecasts can
    reproduce the aggregate histogram while being wrong pixel by pixel.

    `n_bins` defaults to config.FIG5_PIT_BINS. It is worth keeping coarse. The
    pixels are strongly spatially correlated, so a 512x512 tile carries closer
    to one draw's worth of information than to 260,000; the effective sample
    size tracks the tile count, and a fine histogram resolves wiggles the sample
    cannot support. The departures visible at 7 bins are not sampling noise -
    they survive going from 48 tiles to 200, and are unchanged to three decimals
    if the quantile function is interpolated in normal-score space instead of
    linearly in u.
    """
    if n_bins is None:
        n_bins = config.FIG5_PIT_BINS

    pit, w = acc['pit'], acc['pit_w']
    counts, edges = np.histogram(pit, bins=n_bins, range=(0, 1), weights=w)
    density = counts / counts.sum() * n_bins

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.bar(edges[:-1], density, width=1.0 / n_bins, align='edge',
           color='#5fa0d3', edgecolor='white', linewidth=0.4)
    ax.axhline(1.0, color='#22405c', linestyle='--', linewidth=1.2,
               label='Calibrated (uniform)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(1.6, float(density.max()) * 1.12))
    ax.set_xlabel('PIT:  $F_{pred}(y_{obs})$', fontweight='bold', fontsize=13)
    ax.set_ylabel('Density', fontweight='bold', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper center', fontsize=11, frameon=False)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=config.DPI_PLOT, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path.name}  ({n_bins} bins)')


def fig4_5():
    """Figures 4 and 5, from one pass over the hindcast quantile store.

    Figure 4 is the change histogram; Figure 5 is the PIT calibration
    histogram, which used to be panel (b) of the same figure.

    What is gone. Figure 4 was a hexbin of observed against expected change on
    the equal-area grid, and its whole equal-area load went with it. The
    histogram carries the same comparison on the same sample and adds the two
    things the hexbin could not show side by side: the full predictive
    distribution, and what collapsing it to its mean costs.

    The remaining sample weights each native pixel by its ground area rather
    than warping, because the quantile store is only readable on its native
    grid - the same area correction, applied per pixel.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    acc = _fig4_sample()
    _fig4_plot(acc, OUT / 'fig4_obs_vs_pred_dist.png')
    _fig5_plot(acc, OUT / 'fig5_pit_calibration.png')


def figSX():
    OUT.mkdir(parents=True, exist_ok=True)
    utils.register_coolwarm_cmap()
    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    # load HM data
    ds_obs2000 = rxr.open_rasterio(config.PATHS['hm_2000_aa'], chunks='auto')
    ds_obs2005 = rxr.open_rasterio(config.PATHS['hm_2005_aa'], chunks='auto')
    ds_obs2010 = rxr.open_rasterio(config.PATHS['hm_2010_aa'], chunks='auto')
    ds_obs2015 = rxr.open_rasterio(config.PATHS['hm_2015_aa'], chunks='auto')
    ds_obs2020 = rxr.open_rasterio(config.PATHS['hm_2020_aa'], chunks='auto')

    ds_pred2005 = rxr.open_rasterio(config.PATHS['pred_2005_central'], chunks='auto')
    ds_pred2010 = rxr.open_rasterio(config.PATHS['pred_2010_central'], chunks='auto')
    ds_pred2015 = rxr.open_rasterio(config.PATHS['pred_2015_central'], chunks='auto')
    ds_pred2020 = rxr.open_rasterio(config.PATHS['pred_2020_central'], chunks='auto')

    ds_obs = xr.concat([ds_obs2005, ds_obs2010, ds_obs2015, ds_obs2020], dim='time')
    ds_pred = xr.concat([ds_pred2005, ds_pred2010, ds_pred2015, ds_pred2020], dim='time')

    ds_obsdiff = ds_obs - ds_obs2000
    ds_preddiff = ds_pred - ds_obs2000

    ds_diff = xr.Dataset({
        'obs': ds_obsdiff,
        'pred': ds_preddiff
    })
    ds_diff = ds_diff.assign_coords(time=pd.to_datetime([2005, 2010, 2015, 2020], format='%Y'))
    print(ds_diff)

    # ------------------------------------------------------------------
    # Custom colormap + norm
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Figure SX: obs vs pred random (10 random sites)
    # ------------------------------------------------------------------
    #put code here

    # Set random seed for reproducibility
    rng = np.random.default_rng(7)

    # Squeeze band dimension
    hm2000 = ds_obs2000.squeeze('band')
    patch_size = 128
    n_plots = config.FIGSX_N_PLOTS
    sites_per_plot = 2

    # ---------- find good candidate blocks ----------
    hm_valid_frac = hm2000.notnull().coarsen(
        y=patch_size, x=patch_size, boundary='trim'
    ).mean().compute()

    hm_mean = hm2000.coarsen(
        y=patch_size, x=patch_size, boundary='trim'
    ).mean().compute()

    pred_valid_frac = ds_diff['pred'].isel(time=3).squeeze('band').notnull().coarsen(
        y=patch_size, x=patch_size, boundary='trim'
    ).mean().compute()

    obs_abs_change = np.abs(ds_diff['obs'].isel(time=3).squeeze('band')).coarsen(
        y=patch_size, x=patch_size, boundary='trim'
    ).mean().compute()

    candidate_mask = (
        (hm_valid_frac > 0.95).values &
        (hm_mean > 0.05).values &
        (hm_mean < 0.95).values &
        (pred_valid_frac > 0.95).values &
        (obs_abs_change > 0.02).values
    )

    candidate_ys, candidate_xs = np.where(candidate_mask)
    print(f"Found {len(candidate_ys)} candidate blocks")

    total_sites_needed = n_plots * sites_per_plot
    replace = len(candidate_ys) < total_sites_needed
    selected = rng.choice(len(candidate_ys), size=total_sites_needed, replace=replace)
    selected = selected.reshape(n_plots, sites_per_plot)

    years = [2005, 2010, 2015, 2020]

    # Format lat/lon with N/S E/W
    for plot_idx, chosen in enumerate(selected, start=1):
        sites_info = []

        for idx in chosen:
            y0 = int(candidate_ys[idx]) * patch_size
            x0 = int(candidate_xs[idx]) * patch_size
            ysl = slice(y0, y0 + patch_size)
            xsl = slice(x0, x0 + patch_size)

            hm_patch = hm2000.isel(y=ysl, x=xsl).compute().values

            obs_patches, pred_patches = [], []
            for t in range(4):
                obs_p = ds_diff['obs'].isel(time=t).squeeze('band').isel(y=ysl, x=xsl).compute().values
                pred_p = ds_diff['pred'].isel(time=t).squeeze('band').isel(y=ysl, x=xsl).compute().values
                obs_patches.append(np.nan_to_num(obs_p, nan=0.0))
                pred_patches.append(np.nan_to_num(pred_p, nan=0.0))

            lons = hm2000.x.values[x0:x0 + patch_size]
            lats = hm2000.y.values[y0:y0 + patch_size]

            sites_info.append({
                'hm2000': hm_patch,
                'obs_diff': obs_patches,
                'pred_diff': pred_patches,
                'lon_c': float(np.mean(lons)),
                'lat_c': float(np.mean(lats)),
                'extent': [float(lons.min()), float(lons.max()),
                           float(lats.min()), float(lats.max())],
            })

        # Fetch satellite images as plain arrays (no cartopy projection needed)
        print(f"Fetching satellite imagery for plot {plot_idx}...")
        basemap_imgs = [fetch_esri_satellite(s['extent'], size=512) for s in sites_info]
        print("Done.")

        for i, s in enumerate(sites_info):
            lat_s, lon_s = fmt_coord(s['lat_c'], s['lon_c'])
            print(f"Plot {plot_idx} Site {i+1}: {lat_s}, {lon_s}")

        # ---------- build the 5x4 figure ----------
        fig = plt.figure(figsize=(16, 20))

        # Use width_ratios to insert a gap between columns 2 and 3
        # Columns: site1_obs, site1_pred, gap, site2_obs, site2_pred
        gs = fig.add_gridspec(5, 5, hspace=0.12, wspace=0.08,
                              width_ratios=[1, 1, 0.15, 1, 1])

        cmap_hm = 'viridis'
        cmap_diff = plt.get_cmap('my_custom_coolwarm').copy()
        cmap_diff.set_bad(color='#E4E4E4')
        vmin_d, vmax_d = -0.2, 0.2

        im_hm = im_diff = None
        row0_hm_axes = []
        bm_axes = []

        # Map site columns: site 0 -> cols 0,1; site 1 -> cols 3,4 (skip col 2 = gap)
        for s, (site, c0, bm_img) in enumerate(zip(sites_info, [0, 3], basemap_imgs)):
            # Row 0, col c0: observed HM 2000
            ax = fig.add_subplot(gs[0, c0])
            im_hm = ax.imshow(site['hm2000'], cmap=cmap_hm, vmin=0, vmax=1,
                               aspect='auto', interpolation='nearest')
            ax.set_title('Observed HM 2000', fontsize=14, pad=6)
            ax.set_xticks([]); ax.set_yticks([])
            row0_hm_axes.append(ax)

            # Row 0, col c0+1: satellite basemap
            ax_bm = fig.add_subplot(gs[0, c0 + 1])
            ax_bm.imshow(bm_img, aspect='auto', interpolation='bilinear')
            ax_bm.set_title('Satellite imagery', fontsize=14, pad=6)
            ax_bm.set_xticks([]); ax_bm.set_yticks([])
            bm_axes.append(ax_bm)

            # Rows 1-4: observed and predicted HM change
            for r, year in enumerate(years):
                ax_o = fig.add_subplot(gs[r + 1, c0])
                im_diff = ax_o.imshow(site['obs_diff'][r], cmap=cmap_diff,
                                      vmin=vmin_d, vmax=vmax_d,
                                      aspect='auto', interpolation='nearest')
                ax_o.set_xticks([]); ax_o.set_yticks([])

                ax_p = fig.add_subplot(gs[r + 1, c0 + 1])
                ax_p.imshow(site['pred_diff'][r], cmap=cmap_diff,
                            vmin=vmin_d, vmax=vmax_d,
                            aspect='auto', interpolation='nearest')
                ax_p.set_xticks([]); ax_p.set_yticks([])

                # Column headers on the first change row only
                if r == 0:
                    ax_o.set_title('Observed change', fontsize=14, pad=6)
                    ax_p.set_title('Predicted change', fontsize=14, pad=6)

                # Row year labels on the leftmost column only
                if c0 == 0:
                    ax_o.set_ylabel(str(year), fontsize=16, rotation=0, labelpad=30,
                                    va='center', ha='right')

        # ---------- layout adjustments ----------
        fig.subplots_adjust(bottom=0.06, top=0.94)
        fig.canvas.draw()

        # ---------- site labels spanning two columns ----------
        for s, site in enumerate(sites_info):
            lat_s, lon_s = fmt_coord(site['lat_c'], site['lon_c'])
            pos_l = row0_hm_axes[s].get_position()
            pos_r = bm_axes[s].get_position()
            x_center = (pos_l.x0 + pos_r.x1) / 2
            y_top = max(pos_l.y1, pos_r.y1, bm_axes[s].get_position().y1) + 0.015
            fig.text(x_center, y_top,
                     f'Site {s+1}: {lat_s}, {lon_s}',
                     ha='center', va='bottom', fontsize=18, fontweight='bold')

        # ---------- colorbars ----------
        cb1_ax = fig.add_axes([0.08, 0.025, 0.35, 0.012])
        cb1 = fig.colorbar(im_hm, cax=cb1_ax, orientation='horizontal')
        cb1.set_label('HM 2000', fontsize=16)
        cb1.ax.tick_params(labelsize=14)

        cb2_ax = fig.add_axes([0.55, 0.025, 0.35, 0.012])
        cb2 = fig.colorbar(im_diff, cax=cb2_ax, orientation='horizontal')
        cb2.set_label('HM change from 2000', fontsize=16)
        cb2.ax.tick_params(labelsize=14)

        # ---------- globe insets ----------
        for site, ax_bm in zip(sites_info, bm_axes):
            pos = ax_bm.get_position()
            ins = 0.06
            ax_g = fig.add_axes(
                [pos.x1 - ins - 0.005, pos.y1 - ins - 0.005, ins, ins],
                projection=ccrs.Orthographic(
                    central_longitude=site['lon_c'],
                    central_latitude=site['lat_c']
                )
            )
            ax_g.set_global()
            ax_g.stock_img()
            ax_g.coastlines(linewidth=0.3)
            ax_g.plot(site['lon_c'], site['lat_c'], 'ro',
                      markersize=4, transform=ccrs.PlateCarree(), zorder=10)

        output_path = str(OUT / f'fig_SX_obs_vs_predicted_random_{plot_idx:02d}.png')
        fig.savefig(output_path, dpi=config.DPI_MAP, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {output_path}")


def fig6():
    """Figure 6: observed vs predicted change at two sites, with loss risk.

    Each site now gets three columns instead of two. The third carries what the
    old figure could not show: for land that was natural in 2000, the model's
    probability that it has crossed HM >= 0.10 by each year. The central
    prediction in column two answers "how much"; column three answers "how
    likely", and they are different questions - a pixel can have a small
    expected change and still carry a large probability of crossing the line.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    utils.register_coolwarm_cmap()
    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    # load HM data
    ds_obs2000 = rxr.open_rasterio(config.PATHS['hm_2000_aa'], chunks='auto')
    ds_obs2005 = rxr.open_rasterio(config.PATHS['hm_2005_aa'], chunks='auto')
    ds_obs2010 = rxr.open_rasterio(config.PATHS['hm_2010_aa'], chunks='auto')
    ds_obs2015 = rxr.open_rasterio(config.PATHS['hm_2015_aa'], chunks='auto')
    ds_obs2020 = rxr.open_rasterio(config.PATHS['hm_2020_aa'], chunks='auto')

    ds_pred2005 = rxr.open_rasterio(config.PATHS['pred_2005_central'], chunks='auto')
    ds_pred2010 = rxr.open_rasterio(config.PATHS['pred_2010_central'], chunks='auto')
    ds_pred2015 = rxr.open_rasterio(config.PATHS['pred_2015_central'], chunks='auto')
    ds_pred2020 = rxr.open_rasterio(config.PATHS['pred_2020_central'], chunks='auto')

    ds_obs = xr.concat([ds_obs2005, ds_obs2010, ds_obs2015, ds_obs2020], dim='time')
    ds_pred = xr.concat([ds_pred2005, ds_pred2010, ds_pred2015, ds_pred2020], dim='time')

    ds_obsdiff = ds_obs - ds_obs2000
    ds_preddiff = ds_pred - ds_obs2000

    ds_diff = xr.Dataset({
        'obs': ds_obsdiff,
        'pred': ds_preddiff
    })
    ds_diff = ds_diff.assign_coords(time=pd.to_datetime([2005, 2010, 2015, 2020], format='%Y'))

    years = list(config.HINDCAST_YEARS)
    # The hindcast probability rasters are on the same grid but were written by
    # a different tool, so they are only ever indexed positionally here - never
    # combined with the HM arrays through xarray, which would silently
    # inner-join on coordinates (see utils.align_like).
    ds_p10 = [rxr.open_rasterio(config.PATHS[f'hm_p10_{y}'], chunks='auto',
                                masked=True).squeeze(drop=True) for y in years]

    # ------------------------------------------------------------------
    # Figure 6: obs vs predicted, predetermined sites
    # ------------------------------------------------------------------
    hm2000 = ds_obs2000.squeeze('band')
    patch_size = 128

    # ---------- predetermined site centres ----------
    site_coords = [
        (-10.1163, 32.1712),   # Site 1: Muchinga, Zambia
        (-26.02, -61.92),      # Site 2: Gran Chaco, Argentina
    ]

    sites_info = []

    for lat_c, lon_c in site_coords:
        # Find the nearest pixel indices for the centre coordinate
        y_idx = int(np.abs(hm2000.y.values - lat_c).argmin())
        x_idx = int(np.abs(hm2000.x.values - lon_c).argmin())

        # Centre the 128x128 patch on that pixel
        y0 = max(0, y_idx - patch_size // 2)
        x0 = max(0, x_idx - patch_size // 2)
        # Clamp to array bounds
        y0 = min(y0, len(hm2000.y) - patch_size)
        x0 = min(x0, len(hm2000.x) - patch_size)

        ysl = slice(y0, y0 + patch_size)
        xsl = slice(x0, x0 + patch_size)

        hm_patch = hm2000.isel(y=ysl, x=xsl).compute().values
        # Natural in 2000 is the base for every probability panel in this
        # column. The cut is closed at the top - HM <= 0.10, see config.LOW_CUT.
        natural = np.isfinite(hm_patch) & (hm_patch <= config.LOW_CUT)

        obs_patches, pred_patches, p10_patches = [], [], []
        for t in range(4):
            obs_p = ds_diff['obs'].isel(time=t).squeeze('band').isel(y=ysl, x=xsl).compute().values
            pred_p = ds_diff['pred'].isel(time=t).squeeze('band').isel(y=ysl, x=xsl).compute().values
            obs_patches.append(np.nan_to_num(obs_p, nan=0.0))
            pred_patches.append(np.nan_to_num(pred_p, nan=0.0))
            p = ds_p10[t].isel(y=ysl, x=xsl).compute().values
            p10_patches.append(np.where(natural, p, np.nan))

        lons = hm2000.x.values[x0:x0 + patch_size]
        lats = hm2000.y.values[y0:y0 + patch_size]

        sites_info.append({
            'hm2000': hm_patch,
            'natural': natural,
            'obs_diff': obs_patches,
            'pred_diff': pred_patches,
            'p10': p10_patches,
            'lon_c': float(np.mean(lons)),
            'lat_c': float(np.mean(lats)),
            'extent': [float(lons.min()), float(lons.max()),
                       float(lats.min()), float(lats.max())],
        })

    # Fetch satellite images as plain arrays
    print("Fetching satellite imagery...")
    basemap_imgs = [fetch_esri_satellite(s['extent'], size=512) for s in sites_info]
    print("Done.")

    # Format lat/lon with N/S E/W
    predetermined_labels = ['Muchinga, Zambia', 'Chaco, Argentina']
    for i, s in enumerate(sites_info):
        lat_s, lon_s = fmt_coord(s['lat_c'], s['lon_c'])
        print(f"{predetermined_labels[i]}: {lat_s}, {lon_s}")

    # ---------- build the 5x6 figure ----------
    fig = plt.figure(figsize=(21, 19))

    # Columns: site1 x3, gap, site2 x3
    gs = fig.add_gridspec(5, 7, hspace=0.12, wspace=0.08,
                          width_ratios=[1, 1, 1, 0.15, 1, 1, 1])

    cmap_hm = 'viridis'
    cmap_diff = plt.get_cmap('my_custom_coolwarm').copy()
    cmap_diff.set_bad(color='#E4E4E4')
    vmin_d, vmax_d = -0.2, 0.2
    cmap_p, norm_p = utils.exceedance_cmap_norm()
    cmap_p = cmap_p.copy()
    # Out of base within the patch reads as the same beige the global maps use.
    cmap_p.set_bad(color=config.OUT_OF_BASE_HEX)
    # Natural / not natural in 2000, coloured so the beige marks exactly the
    # pixels that are beige in the probability panels below it.
    cmap_natural = mcolors.ListedColormap([config.OUT_OF_BASE_HEX, '#8fbf9f'])

    im_hm = im_diff = im_p = None
    row0_hm_axes = []
    bm_axes = []
    row0_last_axes = []          # third column, for centring the site label

    # site 0 -> cols 0,1,2; site 1 -> cols 4,5,6 (col 3 = gap)
    for s, (site, c0, bm_img) in enumerate(zip(sites_info, [0, 4], basemap_imgs)):
        # Row 0, col c0: observed HM 2000
        ax = fig.add_subplot(gs[0, c0])
        im_hm = ax.imshow(site['hm2000'], cmap=cmap_hm, vmin=0, vmax=1,
                          aspect='auto', interpolation='nearest')
        ax.set_title('Observed HM 2000', fontsize=14, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        row0_hm_axes.append(ax)

        # Row 0, col c0+1: satellite basemap
        ax_bm = fig.add_subplot(gs[0, c0 + 1])
        ax_bm.imshow(bm_img, aspect='auto', interpolation='bilinear')
        ax_bm.set_title('Satellite imagery', fontsize=14, pad=6)
        ax_bm.set_xticks([]); ax_bm.set_yticks([])
        bm_axes.append(ax_bm)

        # Row 0, col c0+2: natural / not natural in 2000
        ax_i = fig.add_subplot(gs[0, c0 + 2])
        ax_i.imshow(site['natural'].astype(float), cmap=cmap_natural,
                    vmin=0, vmax=1, aspect='auto', interpolation='nearest')
        ax_i.set_title(f'Natural in 2000  (HM $\\leq$ {config.LOW_CUT:g})',
                       fontsize=14, pad=6)
        ax_i.set_xticks([]); ax_i.set_yticks([])
        row0_last_axes.append(ax_i)

        # Rows 1-4: observed change, predicted change, probability of loss
        for r, year in enumerate(years):
            ax_o = fig.add_subplot(gs[r + 1, c0])
            im_diff = ax_o.imshow(site['obs_diff'][r], cmap=cmap_diff,
                                  vmin=vmin_d, vmax=vmax_d,
                                  aspect='auto', interpolation='nearest')
            ax_o.set_xticks([]); ax_o.set_yticks([])

            ax_p = fig.add_subplot(gs[r + 1, c0 + 1])
            ax_p.imshow(site['pred_diff'][r], cmap=cmap_diff,
                        vmin=vmin_d, vmax=vmax_d,
                        aspect='auto', interpolation='nearest')
            ax_p.set_xticks([]); ax_p.set_yticks([])

            ax_r = fig.add_subplot(gs[r + 1, c0 + 2])
            im_p = ax_r.imshow(site['p10'][r], cmap=cmap_p, norm=norm_p,
                               aspect='auto', interpolation='nearest')
            ax_r.set_xticks([]); ax_r.set_yticks([])

            # Column headers on the first change row only
            if r == 0:
                ax_o.set_title('Observed change', fontsize=14, pad=6)
                ax_p.set_title('Predicted change', fontsize=14, pad=6)
                ax_r.set_title(f'P(HM > {config.LOW_CUT:g})',
                               fontsize=14, pad=6)

            # Row year labels on the leftmost column only
            if c0 == 0:
                ax_o.set_ylabel(str(year), fontsize=16, rotation=0, labelpad=30,
                                va='center', ha='right')

    # ---------- layout adjustments ----------
    fig.subplots_adjust(bottom=0.06, top=0.94)
    fig.canvas.draw()

    # ---------- site labels spanning the three columns ----------
    for s, site in enumerate(sites_info):
        lat_s, lon_s = fmt_coord(site['lat_c'], site['lon_c'])
        pos_l = row0_hm_axes[s].get_position()
        pos_r = row0_last_axes[s].get_position()
        x_center = (pos_l.x0 + pos_r.x1) / 2
        y_top = max(pos_l.y1, pos_r.y1) + 0.015
        fig.text(x_center, y_top,
                 f'{predetermined_labels[s]}: {lat_s}, {lon_s}',
                 ha='center', va='bottom', fontsize=18, fontweight='bold')

    # ---------- colorbars ----------
    cb1_ax = fig.add_axes([0.07, 0.025, 0.24, 0.012])
    cb1 = fig.colorbar(im_hm, cax=cb1_ax, orientation='horizontal')
    cb1.set_label('HM 2000', fontsize=16)
    cb1.ax.tick_params(labelsize=13)

    cb2_ax = fig.add_axes([0.38, 0.025, 0.24, 0.012])
    cb2 = fig.colorbar(im_diff, cax=cb2_ax, orientation='horizontal')
    cb2.set_label('HM change from 2000', fontsize=16)
    cb2.ax.tick_params(labelsize=13)

    cb3_ax = fig.add_axes([0.69, 0.025, 0.24, 0.012])
    cb3 = fig.colorbar(im_p, cax=cb3_ax, orientation='horizontal',
                       ticks=config.P_EXCEED_LEVELS, spacing='uniform')
    cb3.set_label(f'P(HM > {config.LOW_CUT:g}) | natural in 2000',
                  fontsize=16)
    cb3.ax.set_xticklabels([f'{t:g}' for t in config.P_EXCEED_LEVELS])
    cb3.ax.tick_params(labelsize=11)

    # ---------- globe insets ----------
    for site, ax_bm in zip(sites_info, bm_axes):
        pos = ax_bm.get_position()
        ins = 0.05
        ax_g = fig.add_axes(
            [pos.x1 - ins - 0.005, pos.y1 - ins - 0.005, ins, ins],
            projection=ccrs.Orthographic(
                central_longitude=site['lon_c'],
                central_latitude=site['lat_c']
            )
        )
        ax_g.set_global()
        ax_g.stock_img()
        ax_g.coastlines(linewidth=0.3)
        ax_g.plot(site['lon_c'], site['lat_c'], 'ro',
                  markersize=4, transform=ccrs.PlateCarree(), zorder=10)

    fig.savefig(str(OUT / 'fig_6_obs_vs_predicted_predetermined.png'),
                dpi=config.DPI_MAP, bbox_inches='tight')
    plt.close(fig)
    print('  saved fig_6_obs_vs_predicted_predetermined.png')


def fig7():
    """Figure 7: HM trajectories at four pixels, as full predictive distributions.

    Three rows per pixel. The location map is unchanged. The time series
    replaces the old lower/central/upper ribbon with the continuous quantile
    fan - the ribbon showed three levels of a 64-level distribution and drew
    the eye to its edges, which are the least well determined part of it. The
    third row is the predictive density for 2020 on its own axis, with the 2000
    starting point and the 2020 outcome marked, so the reader can see where the
    truth landed inside the distribution rather than only whether it was
    bracketed.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    # ------------------------------------------------------------------
    # Data prep
    # ------------------------------------------------------------------
    obs_keys = ['hm_1990_aa', 'hm_1995_aa', 'hm_2000_aa', 'hm_2005_aa',
                'hm_2010_aa', 'hm_2015_aa', 'hm_2020_aa']
    obs_years = np.array([1990, 1995, 2000, 2005, 2010, 2015, 2020], dtype=float)
    ds_obs_layers = [rxr.open_rasterio(config.PATHS[k], chunks='auto',
                                       masked=True).squeeze(drop=True)
                     for k in obs_keys]
    hm2000 = ds_obs_layers[2]

    ds_qf = quantiles.open_qf('hindcast')
    qf_times = np.asarray(ds_qf['time'].values, dtype=float)

    # ------------------------------------------------------------------
    # Figure 7: timeseries
    # ------------------------------------------------------------------

    # ---------- setup ----------
    patch_size = 128

    site_coords = [
        (-10.1163, 32.1712),   # Site 1: Muchinga, Zambia
        (-26.02, -61.92),      # Site 2: Gran Chaco, Argentina
    ]
    # ---------- build patches ----------
    site_patches = []
    for lat_c, lon_c in site_coords:
        y_idx = int(np.abs(hm2000.y.values - lat_c).argmin())
        x_idx = int(np.abs(hm2000.x.values - lon_c).argmin())
        y0 = max(0, y_idx - patch_size // 2)
        x0 = max(0, x_idx - patch_size // 2)
        y0 = min(y0, len(hm2000.y) - patch_size)
        x0 = min(x0, len(hm2000.x) - patch_size)

        lons = hm2000.x.values[x0:x0 + patch_size]
        lats = hm2000.y.values[y0:y0 + patch_size]

        site_patches.append({
            'y0': y0, 'x0': x0,
            'lons': lons, 'lats': lats,
            'lat_c': float(np.mean(lats)),
            'lon_c': float(np.mean(lons)),
            'extent': [float(lons.min()), float(lons.max()),
                       float(lats.min()), float(lats.max())],
        })

    # Fetch satellite basemaps once
    print("Fetching satellite imagery...")
    basemap_imgs = [fetch_esri_satellite(s['extent'], size=512) for s in site_patches]
    print("Done.")

    def pixel_record(site_idx, py, px):
        """Observations and the quantile function at one pixel of one patch."""
        site = site_patches[site_idx]
        abs_y = site['y0'] + py
        abs_x = site['x0'] + px
        lat = float(hm2000.y.values[abs_y])
        lon = float(hm2000.x.values[abs_x])
        observed = np.array([float(layer.isel(y=abs_y, x=abs_x).values)
                             for layer in ds_obs_layers])
        # sel(method='nearest') on the store rather than positional indexing:
        # the store carries its own coordinates and we want them to agree with
        # the HM grid on ground position, not on array index.
        p, v_t, _, _ = quantiles.pixel_qf(ds_qf, lat, lon)
        return {'site_idx': site_idx, 'py': py, 'px': px, 'lat': lat, 'lon': lon,
                'observed': observed, 'p': p, 'v': v_t.T}      # v: (n_q, n_time)

    # ---------- iterate seeds ----------
    for seed in config.FIG7_SEEDS:
        rng_site1 = np.random.default_rng(1000 + seed)
        rng_site2 = np.random.default_rng(seed)

        pixel_data = []
        for site_idx, rng in ((0, rng_site1), (1, rng_site2)):
            for _ in range(2):
                py = int(rng.integers(0, patch_size))
                px = int(rng.integers(0, patch_size))
                pixel_data.append(pixel_record(site_idx, py, px))

        print(f"\n--- Seed {seed} ---")

        # ---------- build figure ----------
        fig = plt.figure(figsize=(20, 13))
        gs = fig.add_gridspec(3, 5, hspace=0.34, wspace=0.10,
                              width_ratios=[1, 1, 0.15, 1, 1],
                              height_ratios=[1, 1.3, 1.0])

        col_map = [0, 1, 3, 4]
        predetermined_labels = ['Muchinga, Zambia', 'Chaco, Argentina']
        fan_rgb = mcolors.to_rgb(config.FAN_COLOR)

        for i, (pxd, col) in enumerate(zip(pixel_data, col_map)):
            s_idx = pxd['site_idx']
            bm_img = basemap_imgs[s_idx]
            site = site_patches[s_idx]
            p, v = pxd['p'], pxd['v']
            observed = pxd['observed']

            # ---- Row 0: location map ----
            ax_loc = fig.add_subplot(gs[0, col])
            ax_loc.imshow(bm_img, aspect='auto', interpolation='bilinear')

            scale = 512 / patch_size
            ax_loc.plot(pxd['px'] * scale, pxd['py'] * scale, 'ro', markersize=10,
                        markeredgecolor='white', markeredgewidth=1.5, zorder=10)
            ax_loc.set_xticks([]); ax_loc.set_yticks([])
            lat_s, lon_s = fmt_coord(pxd['lat'], pxd['lon'])
            if seed == 0:
                title_str = f'{predetermined_labels[s_idx]}: {lat_s}, {lon_s}'
            else:
                title_str = f'{lat_s}, {lon_s}'
            ax_loc.set_title(title_str, fontsize=12, pad=6)

            # Globe inset
            pos = ax_loc.get_position()
            ins = 0.075
            ax_g = fig.add_axes(
                [pos.x1 - ins - 0.003, pos.y1 - ins - 0.003, ins, ins],
                projection=ccrs.Orthographic(
                    central_longitude=site['lon_c'],
                    central_latitude=site['lat_c']
                )
            )
            ax_g.set_global()
            ax_g.stock_img()
            ax_g.coastlines(linewidth=0.3)
            ax_g.plot(pxd['lon'], pxd['lat'], 'ro',
                      markersize=4, transform=ccrs.PlateCarree(), zorder=10)

            # ---- Row 1: quantile fan ----
            ax_ts = fig.add_subplot(gs[1, col])

            # Taper the fan back to the last observation before the first
            # horizon, or it starts already wide and reads as uncertainty about
            # a year the model was given.
            t_a, v_a = quantiles.anchor_to_observation(
                qf_times, v, obs_years, observed)
            t_f, y_grid, A, M = quantiles.fan_image(
                t_a, p, v_a, ny=config.FAN_NY, nt=config.FAN_NT,
                mode='prob', gamma=config.FAN_GAMMA,
                ref_time=float(qf_times[0]))

            rgba = np.zeros(A.shape + (4,))
            rgba[..., :3] = fan_rgb
            rgba[..., 3] = np.where(
                M, config.FAN_ALPHA_MIN
                + (config.FAN_ALPHA_MAX - config.FAN_ALPHA_MIN) * A, 0.0)
            ax_ts.imshow(rgba, extent=[t_f[0], t_f[-1], y_grid[0], y_grid[-1]],
                         origin='lower', aspect='auto',
                         interpolation='bilinear', zorder=1)

            k = int(np.argmin(np.abs(p - 0.5)))
            # Red rather than white: the fan is opaque steelblue at the median,
            # so a white line read as a gap in it, and it needed a grey legend
            # patch to be visible at all. Red separates by hue from both the fan
            # and the black observed line, and works on a plain white legend.
            ax_ts.plot(t_a, v_a[k], color='#d62728', linewidth=1.2,
                       linestyle='--', label='Median forecast', zorder=11)

            fin = np.isfinite(observed)
            ax_ts.plot(obs_years[fin], observed[fin], 'k-', linewidth=1.8,
                       label='Observed', zorder=10)

            ax_ts.set_xlabel('Year', fontsize=15)
            if col in (0, 3):
                ax_ts.set_ylabel('HM', fontsize=15)
            else:
                ax_ts.set_yticks([])
            ax_ts.tick_params(labelsize=13)
            ax_ts.set_xlim(1988, 2022)
            ax_ts.set_ylim(0, 1)
            ax_ts.set_xticks([1990, 1995, 2000, 2005, 2010, 2015, 2020])
            ax_ts.set_xticklabels(['1990', '', '2000', '', '2010', '', '2020'],
                                  fontsize=13)
            if i == 0:
                ax_ts.legend(fontsize=13, loc='upper left', framealpha=0.95)

            # ---- Row 2: predictive density for the final hindcast year ----
            ax_d = fig.add_subplot(gs[2, col])
            v_last = v[:, -1]
            start = observed[2]              # 2000, the base year
            truth = observed[-1]             # 2020, the outcome

            # One fixed window for all four panels. They used to be zoomed
            # individually, which made a narrow distribution look as wide as a
            # broad one; on a shared axis the panels are comparable. The lower
            # edge is below zero so a density piled against the HM floor is not
            # clipped by the spine. Anything above 0.5 is off the axis - the
            # quantile function is still integrated over its whole support, only
            # the view is cropped.
            grid = np.linspace(*config.FIG7_DENSITY_XLIM, 500)
            dens = quantiles.density(p, v_last, grid, smooth=4.0)

            ax_d.fill_between(grid, dens, color=config.FAN_COLOR, alpha=0.45,
                              linewidth=0)
            ax_d.plot(grid, dens, color=config.FAN_COLOR, linewidth=1.4)
            if np.isfinite(start):
                ax_d.axvline(start, color='black', linewidth=1.6,
                             label='Observed 2000')
            if np.isfinite(truth):
                ax_d.axvline(truth, color='#b8432a', linewidth=1.6,
                             linestyle='--', label='Observed 2020')

            ax_d.set_xlim(*config.FIG7_DENSITY_XLIM)
            ax_d.set_ylim(bottom=0)
            ax_d.set_yticks([])
            ax_d.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
            ax_d.set_xlabel('Human Modification', fontsize=15)
            if col in (0, 3):
                ax_d.set_ylabel('Predicted\ndensity 2020', fontsize=13)
            ax_d.tick_params(axis='x', labelsize=13)
            for side in ('left', 'top', 'right'):
                ax_d.spines[side].set_visible(False)
            if i == 0:
                ax_d.legend(fontsize=12, loc='upper right', framealpha=0.9)

        out = OUT / f'fig_7_timeseries_site2seed{seed:02d}.png'
        fig.savefig(str(out), dpi=config.DPI_MAP, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved {out}')

    utils.plot_quantile_fan_legend(OUT / 'fig7_fan_legend.png')
    print(f'  Saved {OUT / "fig7_fan_legend.png"}')
    print(f"\nAll {len(config.FIG7_SEEDS)} iterations complete.")


def fig8_9():
    import contextlib as _contextlib

    OUT.mkdir(parents=True, exist_ok=True)
    utils.register_coolwarm_cmap()

    def _build(equal_area: bool, stack=None):
        """The ESRI/HM/CPI stack, on the native grid or the equal-area one.

        Fig 9's map is rendered from the native build; every *number* (the
        Spearman matrix, Fig 8's panels, and the quantile cut-points that set
        Fig 9's colours) comes from the equal-area build, where one cell is one
        constant patch of ground.
        """
        if equal_area:
            with rasterio.open(config.PATHS['hm_2020_aa']) as ref:
                grid = utils.equal_area_grid(ref)
            get = lambda k: utils.open_equal_area_da(          # noqa: E731
                stack, config.PATHS[k], grid, chunks='auto')
            esri, cpi = get('esri_hm'), get('cpi_hm')
            # The blended "central" surface is E[Q], the mean of the predictive
            # distribution, so HM here is an expected change. ESRI and CPI are
            # single surfaces, which is why this comparison stays a comparison
            # of central tendencies rather than of distributions.
            hm = get('hm_central_2040') - get('hm_2020_aa')
        else:
            esri = rxr.open_rasterio(config.PATHS['esri_hm'], chunks='auto')
            cpi = rxr.open_rasterio(config.PATHS['cpi_hm'], chunks='auto')
            hm = utils.load_hm_diff()

        out = xr.Dataset({"ESRI": esri, "HM": hm, "CPI": cpi})
        out['HM'] = out['HM'].where(abs(out['HM']) <= 1)
        return out.where(out.to_array().notnull().all(dim='variable'))

    ds = _build(equal_area=False)
    print(ds)

    # ------------------------------------------------------------------
    # Sample + Spearman correlation
    # ------------------------------------------------------------------
    # Drawn from the equal-area build, so a uniform draw over cells is a
    # uniform draw over ground. Sampling the native grid uniformly over-weights
    # high latitudes and biases both the correlations and Figure 8's panels.
    with _contextlib.ExitStack() as _stack:
        ds_ea = _build(equal_area=True, stack=_stack)
        dims = ('y', 'x') if {'y', 'x'}.issubset(ds_ea.dims) else ('lat', 'lon')
        stacked = ds_ea.stack(points=dims)

        n = 1000_000
        N = stacked.sizes['points']
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(N, size=min(n, N), replace=False))
        sample = stacked.isel(points=idx).load()

        # Cut-points for Fig 9, fitted here and applied to the native grid.
        fit_values = {v: sample[v].values.ravel() for v in ('ESRI', 'HM', 'CPI')}

    simple = sample.drop_vars([c for c in ('x', 'y', 'points') if c in sample.coords])

    # get variables as rows (long)
    vals = simple.to_dataframe()  # multi-index (points, variable) -> value
    vals = vals.reset_index()  # columns -> variables

    vals = vals.drop(columns=[c for c in ('band', 'spatial_ref', 'points')
                              if c in vals.columns])
    #drop row with any na
    vals = vals.dropna()
    #calc spearman correlation - pinned to the three layers so an added column
    # cannot silently widen the reported matrix
    spearman_matrix = vals[['ESRI', 'HM', 'CPI']].corr(method='spearman')

    print(f"Spearman rank correlation (equal-area sample, n = {len(vals):,}):")
    print(spearman_matrix)

    # ------------------------------------------------------------------
    # Figure 8: esri/hm/cpi hexbin pair matrix
    # ------------------------------------------------------------------

    cols = ['ESRI', 'HM', 'CPI']
    plot_data = vals[cols]

    g = sns.PairGrid(plot_data, height=3.6, corner=True)

    hexbin_artists = []

    def hexbin_plot(x, y, color=None, **kwargs):
        ax = plt.gca()
        hb = ax.hexbin(x, y, gridsize=40, bins='log', cmap='inferno', mincnt=1, linewidths=0)
        hexbin_artists.append(hb)

    def hist_plot(x, color=None, **kwargs):
        ax = plt.gca()
        sns.histplot(x=x, bins=30, color='gray', ax=ax)
        ax.yaxis.set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.tick_params(axis='both', labelsize=16)

    g.map_diag(hist_plot)
    g.map_lower(hexbin_plot)

    limits = {
        'ESRI': (0, 1),
        'CPI':  (0, 1),
        'HM':   (-0.1, 0.3)
    }

    for i, row_var in enumerate(cols):
        for j, col_var in enumerate(cols):
            if j > i:
                continue

            ax = g.axes[i, j]
            if ax is None:
                continue

            ax.set_xlim(limits[col_var])

            if i != j:
                ax.set_ylim(limits[row_var])

            ax.tick_params(axis='both', labelsize=16)

    for i, var in enumerate(cols):
        ax = g.axes[i, i]
        if ax is not None:
            ax.set_ylabel('Pixel count', fontsize=18)
            ax.tick_params(axis='both', labelsize=16)

    for i, row_var in enumerate(cols):
        ax = g.axes[i, 0]
        if ax is not None:
            ax.set_ylabel('Pixel count' if i == 0 else row_var, fontsize=18)

    for j, col_var in enumerate(cols):
        ax = g.axes[len(cols) - 1, j]
        if ax is not None:
            ax.set_xlabel(col_var, fontsize=18)

    plot_axes = [ax for row in g.axes for ax in row if ax is not None]
    if hexbin_artists and plot_axes:
        cbar = g.fig.colorbar(hexbin_artists[0], ax=plot_axes, fraction=0.04, pad=0.03)
        cbar.set_label('Pixel count', fontsize=18)
        cbar.ax.tick_params(labelsize=16)

    g.fig.subplots_adjust(top=0.95, right=0.88, wspace=0.22, hspace=0.22)
    g.fig.suptitle('Variable Comparison Matrix (Lower Triangle)', fontsize=20)

    g.savefig(str(OUT / 'fig8_hexbin_matrix_lower.png'), dpi=config.DPI_PLOT)

    # ------------------------------------------------------------------
    # Figure 9 data prep: beta-transform layers
    # ------------------------------------------------------------------

    # Assuming 'ds' is your xarray Dataset containing 'esri', 'hm', 'cpi'
    # We create a new dataset to hold the transformed variables
    ds_transformed = ds.copy(deep=True)

    variables_to_transform = ['ESRI', 'HM', 'CPI']

    for var in variables_to_transform:
        print(f"Transforming {var}...")
        # Fitted on the equal-area sample, applied to the native array: the
        # colour scale becomes a function of ground area while the map stays
        # on its own grid.
        ds_transformed[var] = transform_xarray_layer(
            ds[var],
            a_param=0.8,
            b_param=3,
            fit_values=fit_values[var],
        )

    # Now ds_transformed contains the beta-distributed data
    # with all original coordinates (lat, lon) preserved.

    ds = ds_transformed
    ds = ds.drop_vars('band') if 'band' in ds.coords or 'band' in ds.dims else ds
    # Opacity cut-points likewise come from the equal-area sample, so the
    # legend's "equal-area bins" claim is true of the rendered map.
    ternary_fit = {v: transform_xarray_layer(
        xr.DataArray(fit_values[v]), a_param=0.8, b_param=3,
        fit_values=fit_values[v]).values for v in variables_to_transform}
    rgb = create_ternary_alpha_array(ds, "ESRI", "HM", "CPI",
                                     fit_values=ternary_fit)

    # ------------------------------------------------------------------
    # Figure 9: ternary RGB global map
    # ------------------------------------------------------------------

    # Figure 8 plot - same style as Figure 1 and 2
    hv.extension('matplotlib')

    # Ocean color as RGB tuple for NaN/no-data pixels (matches OCEAN_RGB constant)
    ocean_rgb = np.array([244/255, 252/255, 255/255])  # #F4FCFF

    # Invalid (NaN) pixels are already routed to OCEAN_RGB inside
    # create_ternary_alpha_array, so no white-sentinel replacement is needed.

    # Create xarray DataArray with RGB data
    rgb_da = xr.DataArray(
        rgb,
        coords={'y': ds.y, 'x': ds.x, 'band': ['R', 'G', 'B']},
        dims=['y', 'x', 'band']
    )

    # Free the full-resolution input layers now that rgb is built; the insets
    # below are sliced straight from rgb_da, so ds data variables are no longer
    # needed.  This keeps the render comfortably within the memory budget.
    ds = ds.drop_vars(['ESRI', 'HM', 'CPI'])

    # Create the main RGB plot with same frame size as Figure 1/2
    rgb_plot = rgb_da.hvplot.rgb(
        x='x', y='y', bands='band',
        frame_width=1000,
        frame_height=700,
        pixel_ratio=6,
        xlabel='Longitude',
        ylabel='Latitude',
        rasterize=True,
        projection=ccrs.Robinson(),
        global_extent=True,
    )

    # Render the HoloViews plot to a matplotlib figure
    fig = hv.render(rgb_plot, backend='matplotlib')
    ax = fig.axes[0]

    # Set the background color to light blue (ocean color)
    ax.set_facecolor('#F4FCFF')

    # Add a thin black border around the map region
    ax.spines['geo'].set_visible(True)
    ax.spines['geo'].set_edgecolor('black')
    ax.spines['geo'].set_linewidth(0.3)

    # Remove the default title
    ax.set_title('')

    # Remove any extra axes (e.g. colorbar) since this is an RGB plot
    for a in list(fig.axes[1:]):
        a.remove()

    # Define coordinate reference systems
    robinson = ccrs.Robinson()
    platecarree = ccrs.PlateCarree()

    # Define inset map locations (same as Figure 1 and 2)
    inset_defs = [
        {'center': (-3.046461, -49.938504), 'anchor': (-30, -105)},   # Para, Brazil
        {'center': (0.232389,  37.375075),  'anchor': (-30, -13)},    # Northern Kenya
        {'center': (9.921023,  77.617712),  'anchor': (-30, 77)},     # Southern India
    ]

    # Set the radius of each circular inset in degrees
    radius_deg = 2.0

    # Set the size of inset axes as a fraction of figure size
    inset_size = 0.085

    # Create each inset map
    for ins in inset_defs:
        # Extract center coordinates (where to zoom in) and anchor coordinates (where to place inset)
        clat, clon = ins['center']
        alat, alon = ins['anchor']

        # Transform anchor point from lat/lon to Robinson projection coordinates
        x_rob, y_rob = robinson.transform_point(alon, alat, platecarree)

        # Convert Robinson data coordinates to display coordinates
        disp = ax.transData.transform([x_rob, y_rob])

        # Convert display coordinates to figure coordinates (0-1 range)
        fx, fy = fig.transFigure.inverted().transform(disp)

        # Create inset axes centered on the anchor point
        ax_ins = fig.add_axes(
            [fx - inset_size / 2, fy - inset_size / 2, inset_size, inset_size],
            projection=platecarree
        )

        # Set inset background to ocean color
        ax_ins.set_facecolor('#F4FCFF')

        # Set the geographic extent of the inset (zoom level)
        ax_ins.set_extent(
            [clon - radius_deg, clon + radius_deg, clat - radius_deg, clat + radius_deg],
            crs=platecarree
        )

        # Subset the transformed data for the inset region
        x_sl = slice(clon - radius_deg - 0.1, clon + radius_deg + 0.1)
        y_sl = slice(clat + radius_deg + 0.1, clat - radius_deg - 0.1)

        # Slice the inset straight out of the already-computed RGB (identical pixels
        # to the main map; no recompute, no dependence on the freed ds layers).
        sub_rgb_da = rgb_da.sel(x=x_sl, y=y_sl)
        sub_rgb = sub_rgb_da.values
        sub_x = sub_rgb_da.x.values
        sub_y = sub_rgb_da.y.values

        # Plot the RGB subset in the inset
        ax_ins.imshow(
            sub_rgb,
            extent=[float(sub_x.min()), float(sub_x.max()),
                    float(sub_y.min()), float(sub_y.max())],
            transform=platecarree,
            origin='upper',
            interpolation='nearest'
        )

        # Create a circular boundary path for the inset
        theta = np.linspace(0, 2 * np.pi, 200)
        verts = np.column_stack([
            clon + radius_deg * np.cos(theta),
            clat + radius_deg * np.sin(theta)
        ])
        codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(theta) - 1)
        circle_path = mpath.Path(verts, codes)

        # Apply circular boundary to clip the inset
        ax_ins.set_boundary(circle_path, transform=platecarree)

        # Add a thin black border around the circular inset
        border = Circle((0.5, 0.5), 0.5, transform=ax_ins.transAxes,
                        facecolor='none', edgecolor='black', linewidth=0.3, zorder=6)
        ax_ins.add_patch(border)

        # Remove tick marks from inset
        ax_ins.set_xticks([])
        ax_ins.set_yticks([])

        # Hide inset spines
        for spine in ax_ins.spines.values():
            spine.set_visible(False)

        ax.spines['geo'].set_visible(True)
        ax.spines['geo'].set_edgecolor('black')
        ax.spines['geo'].set_linewidth(0.3)

        # Add a hollow circle marker on the main map showing the inset location
        x_loc, y_loc = robinson.transform_point(clon, clat, platecarree)
        ax.plot(x_loc, y_loc, 'o', color='black', markersize=4,
                markerfacecolor='none', markeredgewidth=0.4,
                transform=ax.transData, zorder=10)

    # Save the figure at high resolution
    fig.savefig(str(OUT / 'fig9_rgb_map.png'), dpi=config.DPI_MAP, bbox_inches='tight')
    # Figure 8 saved to v2/fig8_rgb_map.png

    # ------------------------------------------------------------------
    # Figure 9 legend
    # ------------------------------------------------------------------
    utils.plot_ternary_alpha_legend(OUT / 'fig9_ternary_alpha_legend.png')


def _figS5_S6_map(bound, cbar_label, fname):
    """One global Robinson map of a forecast bound, in the Figure 2 mould."""
    ds = utils.load_hm_diff_bound(bound).to_dataset(name='hm')
    ds = ds.drop('band').squeeze()
    mask = ((ds['hm'] >= -1) & (ds['hm'] <= 1)).compute()
    ds = ds.where(mask, drop=True)
    ds['hm'] = ds['hm'].chunk({'x': 1024, 'y': 1024})
    pds = ds['hm'].compute()

    fig, ax = utils.build_global_robinson_map(
        pds, cmap='my_custom_coolwarm', clim=config.CLIM,
        cbar_label=cbar_label)
    utils.add_circular_insets(fig, ax, pds, cmap='my_custom_coolwarm',
                              vmin=config.CLIM[0], vmax=config.CLIM[1])
    out = OUT / fname
    fig.savefig(out, dpi=config.DPI_MAP, bbox_inches='tight')
    plt.close(fig)
    del ds, pds
    print(f'  saved {out.name}')


def figS5():
    """Figure S5: the lower bound of the 2020-2040 forecast, as a global map.

    Was 11 regional zooms of the central surface - which duplicated Figure 2 at
    a larger scale without adding a quantity. S5 and S6 now carry the two ends
    of the predictive distribution on Figure 2's own scale and insets, so the
    three maps read as one triptych: lower, mean, upper.

    The bound is the 2.5th percentile of each pixel's own distribution, not a
    scenario. See utils.load_hm_diff_bound.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    utils.register_coolwarm_cmap()
    bound, label, fname = config.FIGS_BOUND_VARIANTS[0]
    _figS5_S6_map(bound, label, fname)


def figS6():
    """Figure S6: the upper bound of the 2020-2040 forecast, as a global map.

    Was 22 regional zooms of the two exceedance-probability fields (Figure 3's
    content, region by region). Those probabilities are still in Figures 3a/3b;
    S6 now shows the 97.5th percentile of each pixel's distribution, the
    companion to S5. See figS5.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    utils.register_coolwarm_cmap()
    bound, label, fname = config.FIGS_BOUND_VARIANTS[1]
    _figS5_S6_map(bound, label, fname)



# Risk-class column stems, centre -> rim in the radial and left -> right in
# every legend. Kept in one place so the CSV, the radial and the classifier
# cannot drift apart.
#
# Every area the CSV reports is km2 and says so in its name; the derived shares
# keep the bare stem with a pct_ prefix, since a percentage has no unit.
RISK_KEYS = ('p_low', 'p_mid', 'p_high')
RISK_COLS = {
    'p_low': 'natural_unprot_p_lt025',
    'p_mid': 'natural_unprot_p_025_50',
    'p_high': 'natural_unprot_p_gt50',
}
RISK_AREA_COLS = {k: f'{v}_km2' for k, v in RISK_COLS.items()}
STACK_KEYS = ('protected',) + RISK_KEYS + ('non_natural',)
STACK_PCT = {
    'protected': 'pct_protected',
    'p_low': 'pct_natural_unprot_p_lt025',
    'p_mid': 'pct_natural_unprot_p_025_50',
    'p_high': 'pct_natural_unprot_p_gt50',
    'non_natural': 'pct_unprot_nonnatural_2020',
}


def protection_target_class(pct_protected, pct_low, pct_mid, pct_high,
                            target=None):
    """Risk that an ecoregion cannot reach `target`% protection on natural land.

    The five classes, tested in this order:

      already met                 already at or above the target in 2020
      infeasible on natural land  even all remaining natural land is not enough
      feasible                    the target fits inside land at P(loss) < 0.025
      tight                       it also needs land at 0.025 <= P(loss) < 0.5
      at risk                     it needs land at P(loss) >= 0.5

    `pct_low`, `pct_mid` and `pct_high` are the three risk classes of NATURAL,
    UNPROTECTED land, as percentages of the ecoregion. Kept as a scalar
    function so the boundaries are directly testable; 830 ecoregions is far too
    few for the loop to matter.
    """
    if target is None:
        target = config.PROTECTION_TARGET
    if pct_protected >= target:
        return 'already met'
    if pct_protected + pct_low + pct_mid + pct_high < target:
        return 'infeasible on natural land'
    if pct_protected + pct_low >= target:
        return 'feasible'
    if pct_protected + pct_low + pct_mid >= target:
        return 'tight'
    return 'at risk'


def plot_realm_radial(stats, out_path):
    import matplotlib.patches as mpatches

    """One radial per realm; each arm is one ecoregion, split five ways.

    The five segments partition the ecoregion exactly - protected, the
    three risk classes of unprotected natural land, and land that was
    already non-natural in 2020 - so the arms reach the rim by
    construction. The old version derived its grey band as
    "100 minus everything else", which cannot fail to close and therefore
    checked nothing.
    """
    stats = stats[stats['REALM'] != 'Antarctica'].copy()

    realm_totals = stats.groupby('REALM')['eco_land_km2'].sum().sort_values(ascending=False)
    realms = realm_totals.index.tolist()

    biomes_in_data = (
        stats[['BIOME_NUM', 'BIOME_NAME']]
        .drop_duplicates()
        .sort_values('BIOME_NUM')
        .reset_index(drop=True)
    )

    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 10.5),
                             subplot_kw=dict(projection='polar'),
                             gridspec_kw=dict(hspace=0.05, wspace=0.05))
    axes = axes.flatten()

    r_base = 0.05
    label_radius = 1.12
    for i in range(nrows * ncols):
        ax = axes[i]
        if i >= len(realms):
            ax.set_visible(False)
            continue
        realm = realms[i]
        df_r = stats[stats['REALM'] == realm]
        n_arms = len(df_r)
        if n_arms == 0:
            ax.set_visible(False)
            continue

        biomes_here = sorted(df_r['BIOME_NUM'].unique())
        n_biomes = len(biomes_here)
        n_slots = n_arms + n_biomes  # 1 arm-width gap per biome group
        arm_w = 2.0 * np.pi / n_slots

        cursor = 0.0
        for biome_num in biomes_here:
            sub = df_r[df_r['BIOME_NUM'] == biome_num].sort_values(
                by=['pct_protected', 'eco_land_km2'], ascending=[False, False]
            )
            n = len(sub)
            thetas = cursor + arm_w * (np.arange(n) + 0.5)

            bottom = np.full(n, r_base)
            for key in STACK_KEYS:
                h = (sub[STACK_PCT[key]] / 100.0).values
                ax.bar(thetas, h, width=arm_w * 0.95, bottom=bottom,
                       color=config.RADIAL_COLORS[key], linewidth=0,
                       align='center')
                bottom = bottom + h

            # 30%-of-the-bar reference arc, contained within this biome group
            arc_r = r_base + config.PROTECTION_TARGET / 100.0
            arc_theta = np.linspace(cursor, cursor + arm_w * n, 64)
            ax.plot(arc_theta, np.full_like(arc_theta, arc_r),
                    color='black', alpha=0.6, linewidth=1.0, zorder=5)

            theta_centre = cursor + arm_w * n / 2.0
            ax.text(theta_centre, label_radius, str(int(biome_num)),
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    color='black')

            cursor += arm_w * (n + 1)  # +1 arm-width gap to next biome

        ax.set_ylim(0.0, label_radius + 0.1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines['polar'].set_visible(False)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title(f"{realm}", fontsize=12, pad=10)

    for j in range(len(realms), len(axes)):
        axes[j].set_visible(False)

    color_patches = [
        mpatches.Patch(color=config.RADIAL_COLORS[k],
                       label=config.RADIAL_LABELS[k])
        for k in STACK_KEYS
    ] + [plt.Line2D([0], [0], color='black', alpha=0.6, linewidth=1.0,
                    label=f'{config.PROTECTION_TARGET:g}% target')]
    fig.legend(handles=color_patches, loc='lower center', ncol=3,
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.125))

    biome_handles = [
        mpatches.Patch(facecolor='none', edgecolor='none',
                       label=f"{int(b.BIOME_NUM)} - {b.BIOME_NAME}")
        for _, b in biomes_in_data.iterrows()
    ]
    fig.legend(handles=biome_handles, loc='lower center',
               bbox_to_anchor=(0.5, 0.01),
               ncol=3, fontsize=9, frameon=False,
               handlelength=0, handletextpad=0,
               title='Biome key', title_fontsize=10)

    # subplots_adjust rather than tight_layout: polar axes are not tight_layout
    # compatible (matplotlib says so), so the reserved rect was not honoured and
    # the six-entry colour legend rode up over the bottom row of dials. The old
    # four-entry legend fitted on one line and hid the problem.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.235,
                        hspace=0.26, wspace=0.05)
    fig.savefig(out_path, dpi=config.DPI_PLOT, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f"  saved {out_path.name}")


def fig10():
    """Figure 10: the 30% protection target against forecast loss risk.

    Writes two CSVs and two figures.

    What changed. The old version ran the whole analysis twice, once on the
    central 2040 surface and once on the upper, and emitted a map and a radial
    for each. "Upper" was never a scenario: it was the 97.5th percentile of
    each pixel's own distribution, so a map of it showed the area that would be
    lost only if every pixel realised its unlucky outcome together. There is
    now one analysis, and unprotected natural land is split by its probability
    of loss instead of by which surface crossed a threshold.

    Outputs:
      unprotected_loss_stats.csv  per-ecoregion protection and loss-risk shares
      realm_loss_stats.csv        per-ecoregion expected loss by year (Tables S1-S8)
      fig_unprotected_loss_radial.png
      fig_unprotected_loss_map.png
    """
    import gc
    import os
    import matplotlib.patches as mpatches
    import geopandas as gpd
    from rasterio.features import rasterize

    OUT.mkdir(parents=True, exist_ok=True)
    DATA = config.DATA_DIR
    PATH_HM_2020 = config.PATHS['hm_observed_2020']
    PATH_PA = config.PATHS['hm_static_iucn_strict']
    PATH_ECO = DATA / 'Ecoregions2017' / 'Ecoregions2017.shp'

    PROTECTED_VALUE = 1.0
    P_LOW, P_HIGH = config.P_LOSS_BREAKS
    COLOR_OCEAN = config.OCEAN_HEX
    COLOR_LAND_BASE = '#E0E0E0'

    def _analysis_grid():
        """The shared equal-area grid, derived once from the PA raster.

        Every zonal quantity below is a sum over cells on this grid, so on an
        equal-area CRS a sum of probabilities times cell area IS an expected
        area - which is what the CSV's columns and the Tables S1-S8 captions
        claim. On the native EPSG:4326 grid they would be pixel sums,
        over-weighting the poleward end of every elongated ecoregion.
        """
        with rasterio.open(PATH_PA) as ref:
            return utils.equal_area_grid(ref)

    def _load_raster(path, grid, *, clamp=(0.0, 1.0)):
        """Load `path` onto the equal-area analysis grid.

        All fig10 inputs are float32, so the default NaN fill applies; the 0-1
        clamp then also neutralises the +/-3.4e38 nodata these files disagree
        about. Warping here rather than reading natively is also what makes the
        probability rasters line up with the HM base masks - they carry
        coordinates from a different toolchain and would not align natively
        (see utils.align_like).
        """
        with contextlib.ExitStack() as stack:
            da = utils.open_equal_area_da(stack, path, grid)
            if clamp is not None:
                da = da.where((da >= clamp[0]) & (da <= clamp[1]))
            da.load()      # materialise before the VRT closes
        return da

    def _summary(name, mask):
        n = int(mask.sum())
        km2 = n * (config.EQUAL_AREA_RES / 1000.0) ** 2
        unit = f"cells = {km2:,.0f} km2" if config.EQUAL_AREA_CRS else "pixels"
        print(f"  {name}: {n:,} {unit}")

    def plot_target_map(eco_id, class_of, land_mask, x_coords, y_coords,
                        out_path, stride=4):
        """Ecoregions coloured by their risk of missing the 30% target.

        Decimated by striding rather than coarsening: the values are class ids,
        and a mean or a maximum of two class ids is not a class. Ecoregions are
        far larger than four cells, so striding costs nothing visually. The
        stride is applied to the ids *before* they are mapped to classes, which
        also avoids building a second full-grid array.
        """
        land = land_mask[::stride, ::stride]
        cls = class_of[eco_id[::stride, ::stride]]
        y_c = y_coords[::stride]
        # Trim the columns that straddle the antimeridian; cartopy draws those
        # quads across the whole map. See utils.trim_wrapped_columns.
        x_c, cls, land = utils.trim_wrapped_columns(
            x_coords[::stride], cls, land)

        labels = list(config.TARGET_CLASSES)
        cmap = mcolors.ListedColormap(
            [config.TARGET_CLASS_COLORS[c] for c in labels])
        cmap.set_bad(COLOR_OCEAN, alpha=0.0)
        norm = mcolors.BoundaryNorm(np.arange(-0.5, len(labels) + 0.5), cmap.N)

        fig = plt.figure(figsize=(13, 7))
        ax = plt.axes(projection=ccrs.Robinson())
        ax.set_global()
        ax.set_facecolor(COLOR_OCEAN)

        src_crs = (ccrs.epsg(config.EQUAL_AREA_CRS.split(':')[1])
                   if config.EQUAL_AREA_CRS else ccrs.PlateCarree())

        # Land with no ecoregion (non-terrestrial biomes, unmapped coast) still
        # has to read as land rather than as ocean.
        ax.pcolormesh(x_c, y_c, np.where(land, 1.0, np.nan),
                      cmap=mcolors.ListedColormap([COLOR_LAND_BASE]),
                      vmin=0.0, vmax=1.0, transform=src_crs,
                      shading='nearest', rasterized=True, zorder=1)
        # Masked to land. Several Ecoregions2017 polygons wrap the pole, and in
        # a cylindrical equal-area projection a polar cap becomes a band across
        # every longitude - so an unmasked class layer paints a stripe of
        # "feasible" ecoregion straight across the Arctic Ocean. Every number in
        # the CSV is computed on land_mask; the map has to agree with it.
        ax.pcolormesh(x_c, y_c,
                      np.where((cls >= 0) & land, cls, np.nan).astype(np.float32),
                      cmap=cmap, norm=norm, transform=src_crs,
                      shading='nearest', rasterized=True, zorder=2)

        ax.spines['geo'].set_visible(True)
        ax.spines['geo'].set_edgecolor('black')
        ax.spines['geo'].set_linewidth(0.3)

        def legend_label(c):
            """Class description, prefixed by the class name only where needed.

            Three of the five descriptions already read as complete labels
            ("Target met on land with P(loss) < 0.025"), so the prefix only
            restated them in shorthand. The other two do not stand alone.
            """
            d = config.TARGET_CLASS_DESCRIPTIONS[c]
            if c in config.TARGET_CLASS_LABEL_PREFIX:
                return f'{c.capitalize()} - {d}'
            return d[0].upper() + d[1:]

        handles = [
            mpatches.Patch(color=config.TARGET_CLASS_COLORS[c],
                           label=legend_label(c))
            for c in labels
        ]
        ax.legend(handles=handles, loc='lower center', ncol=2, frameon=False,
                  fontsize=9, bbox_to_anchor=(0.5, -0.16))
        fig.suptitle(
            f'Risk of missing a {config.PROTECTION_TARGET:g}% protection '
            f'target on land natural in 2020', fontsize=13, y=0.95)

        fig.savefig(out_path, dpi=config.DPI_MAP, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)
        print(f"  saved {out_path.name}")

    def main():
        print("=== fig10: protection target vs forecast loss risk ===")
        OUT.mkdir(parents=True, exist_ok=True)

        # [0] Equal-area analysis grid, shared by every layer and the ecoregions
        grid = _analysis_grid()
        if grid is not None:
            print(f"[0] Analysis grid {config.EQUAL_AREA_CRS} @ "
                  f"{config.EQUAL_AREA_RES} m -> {grid[2]:,} x {grid[1]:,} cells")

        # [1] Protected areas
        print("[1] Loading protected-area raster...")
        pa_da = _load_raster(PATH_PA, grid)
        transform = pa_da.rio.transform()
        shape = pa_da.shape
        x_coords = pa_da['x'].values
        y_coords = pa_da['y'].values
        protected = (pa_da.values == PROTECTED_VALUE)
        del pa_da
        _summary("protected", protected)

        # [2] HM 2020 -> the three 2020 condition classes
        print("[2] Loading HM 2020...")
        hm_2020 = _load_raster(PATH_HM_2020, grid)
        hm_vals = hm_2020.values
        land_mask = np.isfinite(hm_vals)
        # Closed at the bottom: natural is HM <= 0.10, so the two classes stay
        # disjoint and partition the land with the rest. See config.LOW_CUT.
        natural_2020 = (hm_vals <= config.LOW_CUT) & land_mask
        mid_2020 = ((hm_vals > config.LOW_CUT) & (hm_vals < config.HIGH_CUT)
                    & land_mask)
        del hm_2020, hm_vals
        _summary("land", land_mask)
        _summary("natural_2020", natural_2020)
        _summary("moderately_modified_2020", mid_2020)

        protected_land = protected & land_mask
        natural_unprotected = natural_2020 & ~protected
        unprot_nonnatural_2020 = land_mask & ~protected & ~natural_2020
        _summary("protected_land", protected_land)
        _summary("natural_unprotected", natural_unprotected)
        _summary("unprot_nonnatural_2020", unprot_nonnatural_2020)
        del protected
        gc.collect()

        # [3] Rasterize ecoregions, before any probability layer is resident
        print("[3] Rasterizing ecoregions...")
        eco = gpd.read_file(str(PATH_ECO), encoding='latin1')
        eco = eco[eco['REALM'].notna() & (eco['REALM'].astype(str) != 'N/A')].reset_index(drop=True)
        if config.EQUAL_AREA_CRS:
            # Burn the polygons on the same grid the rasters were warped to, or
            # the zonal ids would not line up with the masks.
            eco = eco.to_crs(config.EQUAL_AREA_CRS)
        # 69 of 846 geometries are invalid in the source shapefile (the count is
        # identical before and after reprojection); repair them so rasterize
        # cannot silently drop or mis-burn a ring.
        eco['geometry'] = eco.geometry.make_valid()
        eco['idx'] = np.arange(1, len(eco) + 1, dtype=np.uint16)
        print(f"  kept {len(eco)} ecoregions across {eco['REALM'].nunique()} realms")
        shapes_iter = ((g, int(i)) for g, i in zip(eco.geometry, eco['idx']))
        eco_id = rasterize(shapes_iter, out_shape=shape, transform=transform,
                           fill=0, dtype='uint16', all_touched=False)
        N = len(eco)
        gc.collect()

        def zonal_sum(values, chunk_rows=2048):
            """Sum `values` within each ecoregion.

            Row-chunked because np.bincount promotes its weights to float64: a
            whole-grid call would materialise a 4 GB temporary on top of
            everything already resident.
            """
            out = np.zeros(N + 1, dtype=np.float64)
            for r0 in range(0, values.shape[0], chunk_rows):
                sl = slice(r0, min(r0 + chunk_rows, values.shape[0]))
                out += np.bincount(
                    eco_id[sl].ravel(),
                    weights=np.asarray(values[sl], dtype=np.float64).ravel(),
                    minlength=N + 1)
            return out[1:]

        def prob_weights(p, base):
            """p where the base admits it and the layer resolves, else 0."""
            return np.where(base & np.isfinite(p), p, 0.0)

        # [4] Loss-risk classes on unprotected natural land, from p10 in the
        #     final forecast year. This is the layer the protection target is
        #     judged against.
        last = config.FORECAST_YEARS[-1]
        print(f"[4] Loading P(HM >= {config.LOW_CUT:g}) for {last}...")
        p10_last = _load_raster(config.PATHS[f'hm_p10_{last}'], grid)
        p10_vals = p10_last.values
        del p10_last

        resolved = np.isfinite(p10_vals)
        risk = {
            'p_low': natural_unprotected & resolved & (p10_vals < P_LOW),
            'p_mid': natural_unprotected & resolved & (p10_vals >= P_LOW)
                     & (p10_vals < P_HIGH),
            'p_high': natural_unprotected & resolved & (p10_vals >= P_HIGH),
        }
        # Cells the probability layer cannot resolve are reported, not absorbed
        # into a risk class, or the closure check below would be meaningless.
        natural_unprot_nodata = natural_unprotected & ~resolved
        for k in RISK_KEYS:
            _summary(k, risk[k])
        _summary("natural_unprot_nodata", natural_unprot_nodata)

        print("[5] Zonal sums...")
        # Every zonal sum below is a count of analysis cells, and on the
        # equal-area grid one cell is one constant patch of ground - so
        # multiplying by the cell area turns each into km2 here, once, and every
        # area this function reports is km2 from this point on. The percentages
        # further down are ratios of these, so the scaling cancels out of them.
        cell_km2 = (config.EQUAL_AREA_RES / 1000.0) ** 2
        if not config.EQUAL_AREA_CRS:
            print("  WARNING: EQUAL_AREA_CRS is None, so these sums are native "
                  "EPSG:4326 pixel counts and the _km2 columns are mislabelled.")

        def zonal_km2(values):
            return zonal_sum(values) * cell_km2

        columns = {
            'idx': eco['idx'].values,
            'REALM': eco['REALM'].values,
            'ECO_NAME': eco['ECO_NAME'].values,
            'BIOME_NUM': eco['BIOME_NUM'].astype(int).values,
            'BIOME_NAME': eco['BIOME_NAME'].values,
            'eco_land_km2': zonal_km2(land_mask),
            'protected_km2': zonal_km2(protected_land),
            'natural_km2': zonal_km2(natural_2020),
            'unprot_nonnatural_2020_km2': zonal_km2(unprot_nonnatural_2020),
            'natural_unprot_nodata_km2': zonal_km2(natural_unprot_nodata),
        }
        for k in RISK_KEYS:
            columns[RISK_AREA_COLS[k]] = zonal_km2(risk[k])

        # Expected loss by year, for the realm tables. Natural lands lost uses
        # p10 over land natural in 2020; moderately modified lands lost uses p40
        # over land between the two cuts. Both are sums of probabilities, so
        # both are expected areas.
        loss = {}
        loss[f'natural_lost_{last}'] = zonal_km2(prob_weights(p10_vals, natural_2020))
        del p10_vals, risk, resolved
        gc.collect()

        for year in config.FORECAST_YEARS:
            if year != last:
                print(f"[5] P(HM >= {config.LOW_CUT:g}) {year}...")
                p = _load_raster(config.PATHS[f'hm_p10_{year}'], grid).values
                loss[f'natural_lost_{year}'] = zonal_km2(prob_weights(p, natural_2020))
                del p
                gc.collect()
            print(f"[5] P(HM >= {config.HIGH_CUT:g}) {year}...")
            p = _load_raster(config.PATHS[f'hm_p40_{year}'], grid).values
            loss[f'midmod_lost_{year}'] = zonal_km2(prob_weights(p, mid_2020))
            del p
            gc.collect()

        stats = pd.DataFrame(columns)
        loss_df = pd.DataFrame(loss)

        # [6] Target-risk map, before the big masks are released
        skip_maps = os.environ.get('SKIP_MAPS') == '1'

        del (natural_2020, mid_2020, protected_land, natural_unprotected,
             unprot_nonnatural_2020, natural_unprot_nodata)
        gc.collect()

        # Drop non-terrestrial biomes (BIOME_NUM 98=Lake, 99=Rock & Ice, etc.)
        keep = stats['BIOME_NUM'].between(1, 14) & (stats['eco_land_km2'] > 0)
        loss_df = loss_df[keep.values].copy()
        stats = stats[keep].copy()
        print(f"  {len(stats)} ecoregions across "
              f"{stats['REALM'].nunique()} realms and "
              f"{stats['BIOME_NUM'].nunique()} biomes")

        # All percentages are of total ecoregion land area. Every column they
        # divide is already km2 on the equal-area grid, so these are true area
        # shares - and identical to what the old cell-count ratios gave, since
        # the cell area cancels.
        total = stats['eco_land_km2']
        stats['pct_protected'] = 100.0 * stats['protected_km2'] / total
        stats['pct_unprot_nonnatural_2020'] = (
            100.0 * stats['unprot_nonnatural_2020_km2'] / total)
        for k in RISK_KEYS:
            stats[f'pct_{RISK_COLS[k]}'] = (
                100.0 * stats[RISK_AREA_COLS[k]] / total)
        stats['pct_natural_unprot_nodata'] = (
            100.0 * stats['natural_unprot_nodata_km2'] / total)

        # The five stacked shares plus the unresolved remainder partition the
        # ecoregion, so they must close to 100. Unlike the old "grey =
        # 100 - everything else" this can actually fail, which is the point.
        closure = (stats[[STACK_PCT[k] for k in STACK_KEYS]].sum(axis=1)
                   + stats['pct_natural_unprot_nodata'])
        assert np.allclose(closure, 100.0, atol=0.01), (
            'ecoregion shares do not close to 100%: '
            f'worst is {float((closure - 100.0).abs().max()):.4f} points')
        nodata_pct = float(stats['pct_natural_unprot_nodata'].max())
        print(f"  closure check passed; largest unresolved share in any "
              f"ecoregion is {nodata_pct:.4f}%")

        stats['target_class'] = [
            protection_target_class(r.pct_protected,
                                    r.pct_natural_unprot_p_lt025,
                                    r.pct_natural_unprot_p_025_50,
                                    r.pct_natural_unprot_p_gt50)
            for r in stats.itertuples()
        ]
        counts = stats['target_class'].value_counts()
        print(f"  {config.PROTECTION_TARGET:g}% target: "
              + ', '.join(f'{c} {counts.get(c, 0)}'
                          for c in config.TARGET_CLASSES))

        csv_cols = [
            'idx', 'REALM', 'ECO_NAME', 'BIOME_NUM', 'BIOME_NAME',
            'eco_land_km2', 'protected_km2', 'natural_km2',
            'unprot_nonnatural_2020_km2',
            *[RISK_AREA_COLS[k] for k in RISK_KEYS],
            'natural_unprot_nodata_km2',
            'pct_protected', 'pct_unprot_nonnatural_2020',
            *[f'pct_{RISK_COLS[k]}' for k in RISK_KEYS],
            'pct_natural_unprot_nodata', 'target_class',
        ]
        stats[[c for c in csv_cols if c in stats.columns]].to_csv(
            OUT / 'unprotected_loss_stats.csv', index=False)
        print("  saved unprotected_loss_stats.csv")

        # Expected areas for the realm tables. These were hectares, which put a
        # whole-ecoregion total at nine digits beside six-digit losses; km2
        # throughout keeps the table readable and matches every other area the
        # repo reports.
        realm = stats[['idx', 'REALM', 'ECO_NAME', 'BIOME_NUM',
                       'BIOME_NAME']].reset_index(drop=True)
        loss_df = loss_df.reset_index(drop=True)
        # Total ecoregion area, carried through so Tables S1-S8 can print it
        # beside the ecoregion name.
        realm['eco_area_km2'] = stats['eco_land_km2'].reset_index(drop=True)
        for year in config.FORECAST_YEARS:
            for stem in ('natural_lost', 'midmod_lost'):
                realm[f'{stem}_{year}_km2'] = loss_df[f'{stem}_{year}']
        realm.to_csv(OUT / 'realm_loss_stats.csv', index=False)
        print("  saved realm_loss_stats.csv")
        print(f"  expected natural lands lost by {last}, all ecoregions: "
              f"{realm[f'natural_lost_{last}_km2'].sum():,.0f} km2")

        # Ecoregions that cannot reach the target without drawing on land the
        # model gives better-than-even odds of losing. These are the ones where
        # the 30% figure and the forecast are in direct conflict, so the list
        # is printed in full rather than only counted.
        at_risk = stats[stats['target_class'] == 'at risk'].sort_values(
            ['REALM', 'ECO_NAME'])
        head = (f"  ecoregions where the "
                f"{config.PROTECTION_TARGET:g}% target needs land with "
                f"P(loss) > {P_HIGH:g}  ({len(at_risk)})")
        print(head)
        print("  " + "-" * (len(head) - 2))
        for r in at_risk.itertuples():
            print(f"    {r.REALM:<12} {r.ECO_NAME:<58} "
                  f"protected {r.pct_protected:5.1f}%  "
                  f"natural unprotected {r.pct_natural_unprot_p_lt025:5.1f}/"
                  f"{r.pct_natural_unprot_p_025_50:5.1f}/"
                  f"{r.pct_natural_unprot_p_gt50:5.1f}% "
                  f"(P<0.025 / 0.025-0.5 / >0.5)")
        if at_risk.empty:
            print("    none")

        # [7] Figures
        if not skip_maps:
            print("[7] Rendering target-risk map...")
            class_of = np.full(N + 1, -1, dtype=np.int16)
            order = {c: i for i, c in enumerate(config.TARGET_CLASSES)}
            class_of[stats['idx'].to_numpy()] = [
                order[c] for c in stats['target_class']]
            plot_target_map(eco_id, class_of, land_mask, x_coords, y_coords,
                            OUT / 'fig_unprotected_loss_map.png')
        else:
            print("[7] SKIP_MAPS=1, skipping target-risk map")

        print("[8] Rendering radial...")
        plot_realm_radial(stats, OUT / 'fig_unprotected_loss_radial.png')

        print("=== done ===")

    main()
