"""Figure-generating functions. Each writes its outputs to config.OUTPUT_DIR.

Bodies are ported from the standalone plot_*.py scripts; the shared global-map /
inset scaffolding (Figs 2, S1-S4) is delegated to utils, while Fig 3's
categorical map and Figs 8/9's RGB map keep their own verbatim bodies.
"""
import numpy as np
import xarray as xr
import rioxarray as rxr
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
import utils
from utils import (fetch_esri_satellite, fmt_coord,
                   transform_xarray_layer, create_ternary_alpha_array)

OUT = config.OUTPUT_DIR


def fig2_3():
    OUT.mkdir(parents=True, exist_ok=True)
    utils.register_coolwarm_cmap()

    # ---- Figure 2 (shared global-map builder) ----
    ds = rxr.open_rasterio(config.PATHS['hm_diff'], chunks='auto')
    ds = ds.to_dataset(name="hm")
    ds = ds.drop('band')
    ds = ds.squeeze()
    mask = ((ds['hm'] >= -1) & (ds['hm'] <= 1)).compute()
    ds = ds.where(mask, drop=True)
    ds['hm'] = ds['hm'].chunk({'x': 1024, 'y': 1024})
    pds = ds['hm'].compute()

    fig, ax = utils.build_global_robinson_map(
        pds, cmap='my_custom_coolwarm', clim=config.CLIM,
        cbar_label='HM change 2040-2020')
    utils.add_circular_insets(fig, ax, pds, cmap='my_custom_coolwarm',
                              vmin=config.CLIM[0], vmax=config.CLIM[1])
    fig.savefig(OUT / 'fig2_hmdiff_map.png', dpi=config.DPI_MAP, bbox_inches='tight')

    # ---- Figure 3 (verbatim categorical map; kept separate per design) ----
    # NOTE: the original notebook's trailing GeoTIFF re-export of raster_classes.tif
    # is dropped here: it was broken (Dataset-in-Dataset) and redundant with the
    # data/raster_classes.tif input. The two figures are unaffected.
    #load data

    upp = rxr.open_rasterio(config.PATHS['hm_upper_2040'], chunks='auto')
    low = rxr.open_rasterio(config.PATHS['hm_lower_2040'], chunks='auto')
    mid = rxr.open_rasterio(config.PATHS['hm_central_2040'], chunks='auto')
    hm = rxr.open_rasterio(config.PATHS['hm_observed_2020'], chunks='auto')

    ds = xr.Dataset({
        "upper": upp,
        "lower": low,
        "central": mid,
        "hm": hm,
    })
    ds = ds.drop_vars("band")
    ds = ds.squeeze()
    ds

    #convert to  0 or 1 using threshold of 0.1
    ds_binary = ds.where(ds > 0.1, 1)
    ds_binary = ds_binary.where(ds_binary <= 0.1, 0)
    ds_binary

    # Create a raster with 5 classes based on the conditions
    raster = xr.zeros_like(ds['hm'])

    # 0 when 'hm' is 1
    raster = raster.where(ds_binary['hm'] != 1, 0)

    # 1 when 'hm' is 0 and 'upper' is 0
    mask_1 = (ds_binary['hm'] == 0) & (ds_binary['upper'] == 0)
    raster = raster.where(~mask_1, 1)

    # 2 when 'hm' is 0 and 'upper' is 1
    mask_2 = (ds_binary['hm'] == 0) & (ds_binary['upper'] == 1)
    raster = raster.where(~mask_2, 2)

    # 3 when 'hm' is 0 and 'central' is 1
    mask_3 = (ds_binary['hm'] == 0) & (ds_binary['central'] == 1)
    raster = raster.where(~mask_3, 3)

    # 4 when 'hm' is 0 and 'lower' is 1
    mask_4 = (ds_binary['hm'] == 0) & (ds_binary['lower'] == 1)
    raster = raster.where(~mask_4, 4)

    #set values of ds['hm'] > 1 to na
    mask_5 = (ds['hm'] > 1)
    raster = raster.where(~mask_5, np.nan)

    mask = xr.where(ds['hm'].notnull(), 0, 10)
    raster = raster + mask
    #raster = raster.where(raster >= 0)

    #raster.to_zarr('output_ras.zarr', mode='w')
    #raster = xr.open_zarr('output_ras.zarr')

    #read raster from data/raster_classes.tif
    raster = rxr.open_rasterio(config.PATHS['raster_classes'])

    #drop band dimension
    raster = raster.squeeze()
    #conver to ds with var hm
    raster = raster.to_dataset(name="hm")
    raster

    raster = raster.where(~ds['upper'].isnull())

    # ------------------------------------------------------------------
    # Figure 3
    # ------------------------------------------------------------------
    #create and export fig2 plot

    # Create and register the same 5-class colormap currently used for Figure 2
    colors = ['#000000', '#bee6c2', '#ffbb00', '#ff0000', '#dd87ff']
    custom_cmap = ListedColormap(colors)

    try:
        plt.colormaps.register(name='raster_classes', cmap=custom_cmap)
    except ValueError:
        plt.colormaps.unregister('raster_classes')
        plt.colormaps.register(name='raster_classes', cmap=custom_cmap)

    # Normalize to a 2D DataArray (works whether raster is DataArray or Dataset)
    if isinstance(raster, xr.Dataset):
        if 'hm' in raster.data_vars:
            raster_da = raster['hm']
        else:
            raster_da = raster[next(iter(raster.data_vars))]
    else:
        raster_da = raster

    # Keep only classes 0-4 for plotting (everything else -> NaN/ocean)
    raster_plot_data = raster_da.where((raster_da >= 0) & (raster_da <= 4))

    # Downsample main map for stability while keeping Figure 1 layout
    main_plot_data = raster_plot_data.isel(x=slice(None, None, 8), y=slice(None, None, 8)).compute()

    # Build main map with Figure 1 plotting/layout settings
    hv.extension('matplotlib')
    plot = main_plot_data.hvplot.quadmesh(
        x='x', y='y',
        frame_width=1000,
        frame_height=700,
        pixel_ratio=6,
        xlabel='Longitude',
        ylabel='Latitude',
        projection=ccrs.Robinson(),
        global_extent=True,
        cmap='raster_classes'
    ).opts(
        clim=(0, 4)
    )

    fig = hv.render(plot, backend='matplotlib')
    ax = fig.axes[0]

    # Match Figure 1 map border and background styling
    ax.set_facecolor('#F4FCFF')
    ax.spines['geo'].set_visible(True)
    ax.spines['geo'].set_edgecolor('black')
    ax.spines['geo'].set_linewidth(0.3)
    ax.set_title('')

    # Match Figure 1 legend size/location, keep categorical text labels
    legend_labels = [
        'Non-natural 2020',
        'Still Natural 2040',
        'Natural lands loss 2040 (upper 97.5% forecast)',
        'Natural lands loss 2040 (central 50% forecasts)',
        'Natural lands loss 2040 (lower 2.5% forecast)'
    ]

    for a in fig.axes[1:]:
        pos = a.get_position()
        a.set_position([pos.x0, pos.y0 + pos.height * 0.25, pos.width * 0.4, pos.height * 0.3])
        a.tick_params(labelsize=3, width=0.0, length=0)
        for spine in a.spines.values():
            spine.set_linewidth(0.3)
        a.set_yticks([0, 1, 2, 3, 4])
        a.set_yticklabels(legend_labels)
        a.set_ylabel('')

    # Reuse Figure 1 inset layout/locations
    robinson = ccrs.Robinson()
    platecarree = ccrs.PlateCarree()

    inset_cmap = plt.colormaps['raster_classes'].copy()
    inset_cmap.set_bad('#F4FCFF')

    inset_defs = [
        {'center': (-3.046461, -49.938504), 'anchor': (-30, -105)},
        {'center': (0.232389, 37.375075), 'anchor': (-30, -13)},
        {'center': (9.921023, 77.617712), 'anchor': (-30, 77)},
    ]

    radius_deg = 2.0
    inset_size = 0.085

    # Coordinate-order-aware slicing for robust inset extraction
    x0, x1 = float(raster_plot_data.x.values[0]), float(raster_plot_data.x.values[-1])
    y0, y1 = float(raster_plot_data.y.values[0]), float(raster_plot_data.y.values[-1])
    x_ascending = x1 > x0
    y_ascending = y1 > y0

    for ins in inset_defs:
        clat, clon = ins['center']
        alat, alon = ins['anchor']

        x_rob, y_rob = robinson.transform_point(alon, alat, platecarree)
        disp = ax.transData.transform([x_rob, y_rob])
        fx, fy = fig.transFigure.inverted().transform(disp)

        ax_ins = fig.add_axes(
            [fx - inset_size / 2, fy - inset_size / 2, inset_size, inset_size],
            projection=platecarree
        )
        ax_ins.set_facecolor('#F4FCFF')
        ax_ins.set_extent(
            [clon - radius_deg, clon + radius_deg, clat - radius_deg, clat + radius_deg],
            crs=platecarree
        )

        x_min = clon - radius_deg - 0.1
        x_max = clon + radius_deg + 0.1
        y_min = clat - radius_deg - 0.1
        y_max = clat + radius_deg + 0.1

        x_slice = slice(x_min, x_max) if x_ascending else slice(x_max, x_min)
        y_slice = slice(y_min, y_max) if y_ascending else slice(y_max, y_min)

        sub = raster_plot_data.sel(x=x_slice, y=y_slice)

        if {'x', 'y'}.issubset(sub.dims) and sub.sizes.get('x', 0) > 1 and sub.sizes.get('y', 0) > 1:
            sub2d = sub.compute()
            vals = np.asarray(sub2d.values)
            if vals.ndim == 2 and vals.shape[0] > 1 and vals.shape[1] > 1:
                ax_ins.pcolormesh(
                    sub2d.x.values,
                    sub2d.y.values,
                    vals,
                    cmap=inset_cmap,
                    vmin=0,
                    vmax=4,
                    transform=platecarree,
                    shading='auto'
                )

        theta = np.linspace(0, 2 * np.pi, 200)
        verts = np.column_stack([
            clon + radius_deg * np.cos(theta),
            clat + radius_deg * np.sin(theta)
        ])
        codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(theta) - 1)
        circle_path = mpath.Path(verts, codes)
        ax_ins.set_boundary(circle_path, transform=platecarree)

        border = Circle((0.5, 0.5), 0.5, transform=ax_ins.transAxes,
                        facecolor='none', edgecolor='black', linewidth=0.3, zorder=6)
        ax_ins.add_patch(border)

        ax_ins.set_xticks([])
        ax_ins.set_yticks([])
        for spine in ax_ins.spines.values():
            spine.set_visible(False)

        # Same inset-location marker placement/style as Figure 1
        x_loc, y_loc = robinson.transform_point(clon, clat, platecarree)
        ax.plot(
            x_loc,
            y_loc,
            'o',
            color='black',
            markersize=4,
            markerfacecolor='none',
            markeredgewidth=0.4,
            transform=ax.transData,
            zorder=10
        )

    fig.savefig(OUT / 'fig3_uncer_map_hr.png', dpi=config.DPI_MAP, bbox_inches='tight')


def figS1_S4():
    """Fig S1 (observed 2020-2000 diff) + Figs S2/S3/S4 (2025/2030/2035 - 2020).

    S2-S4 were three byte-identical notebook blocks differing only by prediction
    year, colorbar label, and output name; reproduced here as one loop driven by
    config.FIGS_YEAR_VARIANTS.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    utils.register_coolwarm_cmap()

    # ---- Figure S1: observed HM change 2020-2000 ----
    dsobs = rxr.open_rasterio(config.PATHS['hm_diff_obs'], chunks='auto')
    dsobs = dsobs.to_dataset(name="hm")
    ds = dsobs.drop('band')
    ds = ds.squeeze()
    mask = ((ds['hm'] >= -1) & (ds['hm'] <= 1)).compute()
    ds = ds.where(mask, drop=True)
    ds['hm'] = ds['hm'].chunk({'x': 1024, 'y': 1024})
    pds = ds['hm'].compute()

    fig, ax = utils.build_global_robinson_map(
        pds, cmap='my_custom_coolwarm', clim=config.CLIM,
        cbar_label='HM change 2020-2000', hide_geo_spine=True)
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


def fig4_5():
    OUT.mkdir(parents=True, exist_ok=True)
    # ------------------------------------------------------------------
    # Data (shared by Figure 4 and Figure 5)
    # ------------------------------------------------------------------
    # load HM data
    ds_obs2000 = rxr.open_rasterio(config.PATHS['hm_2000_aa'], chunks='auto')
    ds_obs = rxr.open_rasterio(config.PATHS['hm_2020_aa'], chunks='auto')

    ds_pred = rxr.open_rasterio(config.PATHS['pred_2020_central'], chunks='auto')

    ds_obsdiff = ds_obs - ds_obs2000
    ds_preddiff = ds_pred - ds_obs2000

    ds_diff = xr.Dataset({
        'obs': ds_obsdiff,
        'pred': ds_preddiff
    })
    print(ds_diff)

    #lodd mask data
    splitmask = rxr.open_rasterio(config.PATHS['split_mask'])
    print(splitmask)

    #mask to where split == 2 (validation set)
    val_mask = splitmask.isel(band=0) == 2
    ds_diff_masked = ds_diff.where(val_mask, drop=False)

    # Flatten and drop NaNs
    obs_vals = ds_diff_masked['obs'].values.ravel()
    pred_vals = ds_diff_masked['pred'].values.ravel()

    valid = np.isfinite(obs_vals) & np.isfinite(pred_vals)
    obs_flat = obs_vals[valid]
    pred_flat = pred_vals[valid]

    print(obs_flat)
    print(pred_flat)

    # ------------------------------------------------------------------
    # Figure 4: obs vs pred hexbin
    # ------------------------------------------------------------------
    # Clip to plot range
    vmin, vmax = 0.0, 0.2
    mask_range = (obs_flat >= vmin) & (obs_flat <= vmax) & (pred_flat >= vmin) & (pred_flat <= vmax)
    obs_plot = obs_flat[mask_range]
    pred_plot = pred_flat[mask_range]

    # Create figure
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    # 2D histogram with log-scaled colour using cubehelix
    hb = ax.hexbin(
        obs_plot, pred_plot,
        gridsize=150,
        cmap='cubehelix',
        norm=mcolors.LogNorm(vmin=1, vmax=1e6),
        mincnt=1,
        extent=[vmin, vmax, vmin, vmax],
    )

    # 1:1 reference line
    ax.plot(
        [vmin, vmax], [vmin, vmax],
        linestyle='--', color='grey', linewidth=1, label='1:1 line'
    )

    # Colour bar
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label('Count (log scale)', fontsize=14)
    cb.ax.tick_params(labelsize=10)

    # Axis labels – using "change" consistently
    ax.set_xlabel('Observed change', fontsize=14)
    ax.set_ylabel('Modelled change', fontsize=14)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', labelsize=10)
    ax.legend(loc='upper left', fontsize=12)

    plt.tight_layout()
    plt.savefig(str(OUT / 'fig4_obs_vs_pred_change_hexbin.png'), dpi=config.DPI_PLOT, bbox_inches='tight')
    plt.show()

    # ------------------------------------------------------------------
    # Figure 5: obs vs pred hist
    # ------------------------------------------------------------------
    bin_edges = np.array([-1.0, -0.05, 0.0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    bin_labels = [
        '-1 to -0.05',
        '-0.05 to 0',
        '0 to 0.005',
        '0.005 to 0.02',
        '0.02 to 0.05',
        '0.05 to 0.1',
        '0.1 to 0.2',
        '0.2 to 0.5',
        '0.5 to 1',
    ]

    obs_counts, _ = np.histogram(obs_flat, bins=bin_edges)
    pred_counts, _ = np.histogram(pred_flat, bins=bin_edges)

    x = np.arange(len(bin_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.55, 4.36))

    ax.bar(x - width / 2, obs_counts, width, color='#58c785', label='Observed')
    ax.bar(x + width / 2, pred_counts, width, color='#5fa0d3', label='Predicted')

    ax.set_yscale('log')
    ax.set_title('', fontweight='bold')
    ax.set_xlabel('HM Change (2020 - 2000)', fontweight='bold', fontsize=13)
    ax.set_ylabel('Count (log scale)', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=12)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.savefig(str(OUT / 'fig5_obs_vs_pred_hist.png'), dpi=config.DPI_PLOT, bbox_inches='tight')
    plt.show()


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
            ax_bm.set_title('Location', fontsize=14, pad=6)
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
            y_top = max(pos_l.y1, pos_r.y1) + 0.015
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
    # Figure 6: obs vs predicted predetermined sites
    # ------------------------------------------------------------------
    #put code here

    # Squeeze band dimension
    hm2000 = ds_obs2000.squeeze('band')
    patch_size = 128

    # ---------- predetermined site centres ----------
    -11.5920,20.0837
    site_coords = [
        (-10.1163,32.1712),    # Site 1: zambia
        (-26.02, -61.92),  # Site 2: granchaco argentina

    ]

    years = [2005, 2010, 2015, 2020]
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

    # Fetch satellite images as plain arrays
    print("Fetching satellite imagery...")
    basemap_imgs = [fetch_esri_satellite(s['extent'], size=512) for s in sites_info]
    print("Done.")

    # Format lat/lon with N/S E/W
    for i, s in enumerate(sites_info):
        lat_s, lon_s = fmt_coord(s['lat_c'], s['lon_c'])
        print(f"Site {i+1}: {lat_s}, {lon_s}")

    # ---------- build the 5x4 figure ----------
    fig = plt.figure(figsize=(16, 20))

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

    # site 0 -> cols 0,1; site 1 -> cols 3,4 (col 2 = gap)
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
        ax_bm.set_title('Location', fontsize=14, pad=6)
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
        y_top = max(pos_l.y1, pos_r.y1) + 0.015
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

    fig.savefig(str(OUT / 'fig_6_obs_vs_predicted_predetermined'), dpi=config.DPI_MAP, bbox_inches='tight')
    plt.show()


def fig7():
    OUT.mkdir(parents=True, exist_ok=True)
    # ------------------------------------------------------------------
    # Data prep
    # ------------------------------------------------------------------
    # load HM data
    ds_obs1990 = rxr.open_rasterio(config.PATHS['hm_1990_aa'], chunks='auto')
    ds_obs1995 = rxr.open_rasterio(config.PATHS['hm_1995_aa'], chunks='auto')
    ds_obs2000 = rxr.open_rasterio(config.PATHS['hm_2000_aa'], chunks='auto')
    ds_obs2005 = rxr.open_rasterio(config.PATHS['hm_2005_aa'], chunks='auto')
    ds_obs2010 = rxr.open_rasterio(config.PATHS['hm_2010_aa'], chunks='auto')
    ds_obs2015 = rxr.open_rasterio(config.PATHS['hm_2015_aa'], chunks='auto')
    ds_obs2020 = rxr.open_rasterio(config.PATHS['hm_2020_aa'], chunks='auto')

    ds_pred2005_central = rxr.open_rasterio(config.PATHS['pred_2005_central'], chunks='auto')
    ds_pred2005_upper = rxr.open_rasterio(config.PATHS['pred_2005_upper'], chunks='auto')
    ds_pred2005_lower = rxr.open_rasterio(config.PATHS['pred_2005_lower'], chunks='auto')
    ds_pred2010_central = rxr.open_rasterio(config.PATHS['pred_2010_central'], chunks='auto')
    ds_pred2010_upper = rxr.open_rasterio(config.PATHS['pred_2010_upper'], chunks='auto')
    ds_pred2010_lower = rxr.open_rasterio(config.PATHS['pred_2010_lower'], chunks='auto')
    ds_pred2015_central = rxr.open_rasterio(config.PATHS['pred_2015_central'], chunks='auto')
    ds_pred2015_upper = rxr.open_rasterio(config.PATHS['pred_2015_upper'], chunks='auto')
    ds_pred2015_lower = rxr.open_rasterio(config.PATHS['pred_2015_lower'], chunks='auto')
    ds_pred2020_central = rxr.open_rasterio(config.PATHS['pred_2020_central'], chunks='auto')
    ds_pred2020_upper = rxr.open_rasterio(config.PATHS['pred_2020_upper'], chunks='auto')
    ds_pred2020_lower = rxr.open_rasterio(config.PATHS['pred_2020_lower'], chunks='auto')

    full_time = pd.to_datetime([1990, 1995, 2000, 2005, 2010, 2015, 2020], format='%Y')
    pred_time = pd.to_datetime([2005, 2010, 2015, 2020], format='%Y')

    ds_obs = xr.concat(
        [ds_obs1990, ds_obs1995, ds_obs2000, ds_obs2005, ds_obs2010, ds_obs2015, ds_obs2020],
        dim=pd.Index(full_time, name='time')
    )

    ds_pred_central = xr.concat(
        [ds_pred2005_central, ds_pred2010_central, ds_pred2015_central, ds_pred2020_central],
        dim=pd.Index(pred_time, name='time')
    ).reindex(time=full_time)

    ds_pred_upper = xr.concat(
        [ds_pred2005_upper, ds_pred2010_upper, ds_pred2015_upper, ds_pred2020_upper],
        dim=pd.Index(pred_time, name='time')
    ).reindex(time=full_time)

    ds_pred_lower = xr.concat(
        [ds_pred2005_lower, ds_pred2010_lower, ds_pred2015_lower, ds_pred2020_lower],
        dim=pd.Index(pred_time, name='time')
    ).reindex(time=full_time)

    ds_combined = xr.Dataset({
        'observed': ds_obs,
        'central': ds_pred_central,
        'upper': ds_pred_upper,
        'lower': ds_pred_lower
    })
    print(ds_combined)

    # ------------------------------------------------------------------
    # Figure 7: timeseries
    # ------------------------------------------------------------------

    # ---------- setup ----------
    patch_size = 128
    hm2000 = ds_obs2000.squeeze('band')

    site_coords = [
        (-10.1163,32.1712),    # Site 1: zambia
        (-26.02, -61.92),  # Site 2: chaco
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

    time_vals = ds_combined.time.values
    years = [pd.Timestamp(t).year for t in time_vals]

    # ---------- iterate seed 0-20 for site 2 ----------
    for seed in config.FIG7_SEEDS:
        rng_site1 = np.random.default_rng(1000 + seed)
        site_patches[0]['pixels'] = [(int(rng_site1.integers(0, patch_size)), int(rng_site1.integers(0, patch_size)))
                                      for _ in range(2)]

        pixel_data_site1 = []
        for py, px in site_patches[0]['pixels']:
            abs_y = site_patches[0]['y0'] + py
            abs_x = site_patches[0]['x0'] + px
            obs_ts = ds_combined['observed'].isel(band=0, y=abs_y, x=abs_x).compute().values
            cen_ts = ds_combined['central'].isel(band=0, y=abs_y, x=abs_x).compute().values
            upp_ts = ds_combined['upper'].isel(band=0, y=abs_y, x=abs_x).compute().values
            low_ts = ds_combined['lower'].isel(band=0, y=abs_y, x=abs_x).compute().values
            pixel_lat = float(hm2000.y.values[abs_y])
            pixel_lon = float(hm2000.x.values[abs_x])
            pixel_data_site1.append({
                'site_idx': 0, 'py': py, 'px': px,
                'lat': pixel_lat, 'lon': pixel_lon,
                'observed': obs_ts, 'central': cen_ts, 'upper': upp_ts, 'lower': low_ts,
            })

        rng_site2 = np.random.default_rng(seed)
        site_patches[1]['pixels'] = [(int(rng_site2.integers(0, patch_size)), int(rng_site2.integers(0, patch_size)))
                                      for _ in range(2)]

        pixel_data_site2 = []
        for py, px in site_patches[1]['pixels']:
            abs_y = site_patches[1]['y0'] + py
            abs_x = site_patches[1]['x0'] + px
            obs_ts = ds_combined['observed'].isel(band=0, y=abs_y, x=abs_x).compute().values
            cen_ts = ds_combined['central'].isel(band=0, y=abs_y, x=abs_x).compute().values
            upp_ts = ds_combined['upper'].isel(band=0, y=abs_y, x=abs_x).compute().values
            low_ts = ds_combined['lower'].isel(band=0, y=abs_y, x=abs_x).compute().values
            pixel_lat = float(hm2000.y.values[abs_y])
            pixel_lon = float(hm2000.x.values[abs_x])
            pixel_data_site2.append({
                'site_idx': 1, 'py': py, 'px': px,
                'lat': pixel_lat, 'lon': pixel_lon,
                'observed': obs_ts, 'central': cen_ts, 'upper': upp_ts, 'lower': low_ts,
            })

        pixel_data = pixel_data_site1 + pixel_data_site2
        print(f"\n--- Seed {seed} ---")

        # ---------- build figure ----------
        fig = plt.figure(figsize=(20, 9))
        gs = fig.add_gridspec(2, 5, hspace=0.30, wspace=0.10,
                              width_ratios=[1, 1, 0.15, 1, 1],
                              height_ratios=[1, 1.3])

        col_map = [0, 1, 3, 4]

        for i, (pxd, col) in enumerate(zip(pixel_data, col_map)):
            s_idx = pxd['site_idx']
            bm_img = basemap_imgs[s_idx]
            site = site_patches[s_idx]

            # ---- Row 0: location map ----
            ax_loc = fig.add_subplot(gs[0, col])
            ax_loc.imshow(bm_img, aspect='auto', interpolation='bilinear')

            scale = 512 / patch_size
            dot_x = pxd['px'] * scale
            dot_y = pxd['py'] * scale
            ax_loc.plot(dot_x, dot_y, 'ro', markersize=10,
                        markeredgecolor='white', markeredgewidth=1.5, zorder=10)
            ax_loc.set_xticks([]); ax_loc.set_yticks([])
            lat_s, lon_s = fmt_coord(pxd['lat'], pxd['lon'])
            ax_loc.set_title(f'{lat_s}, {lon_s}', fontsize=16, pad=6)

            # Globe inset
            pos = ax_loc.get_position()
            ins = 0.11
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

            # ---- Row 1: time series ----
            ax_ts = fig.add_subplot(gs[1, col])

            valid = ~np.isnan(pxd['central'])
            yrs_valid = np.array(years)[valid]
            ax_ts.fill_between(yrs_valid, pxd['lower'][valid], pxd['upper'][valid],
                                alpha=0.3, color='cornflowerblue', label='95% PI')
            ax_ts.plot(yrs_valid, pxd['central'][valid], 'r--',
                       linewidth=1.8, label='Predicted')

            obs_valid = ~np.isnan(pxd['observed'])
            ax_ts.plot(np.array(years)[obs_valid], pxd['observed'][obs_valid],
                       'k-', linewidth=1.8, label='Observed')

            ax_ts.set_xlabel('Year', fontsize=16)
            if col in (0, 3):
                ax_ts.set_ylabel('HM', fontsize=16)
            else:
                ax_ts.set_yticks([])
            ax_ts.tick_params(labelsize=14)
            ax_ts.set_xlim(1988, 2022)
            ax_ts.set_ylim(0, 1)
            ax_ts.set_xticks([1990, 1995, 2000, 2005, 2010, 2015, 2020])
            ax_ts.set_xticklabels(['1990', '', '2000', '', '2010', '', '2020'], fontsize=14)

            if i == 0:
                ax_ts.legend(fontsize=14, loc='upper left', framealpha=0.9)

        fig.savefig(str(OUT / f'fig_7_timeseries_site2seed{seed:02d}.png'), dpi=config.DPI_MAP, bbox_inches='tight')
        plt.close(fig)
        print("  Saved " + str(OUT / f"fig_7_timeseries_site2seed{seed:02d}.png"))

    print(f"\nAll {len(config.FIG7_SEEDS)} iterations complete.")


def fig8_9():
    OUT.mkdir(parents=True, exist_ok=True)
    utils.register_coolwarm_cmap()
    esri = rxr.open_rasterio(config.PATHS['esri_hm'], chunks='auto')
    cpi = rxr.open_rasterio(config.PATHS['cpi_hm'], chunks='auto')
    hm = rxr.open_rasterio(config.PATHS['hm_diff'], chunks='auto')

    ds = xr.Dataset({
        "esri": esri,
        "hm": hm,
        "cpi": cpi
    })

    ds['hm'] = ds['hm'].where(abs(ds['hm']) <= 1)
    mask = ds.to_array().notnull().all(dim='variable')
    ds = ds.where(mask)
    print(ds)

    # ------------------------------------------------------------------
    # Sample + Spearman correlation
    # ------------------------------------------------------------------

    dims = ('y', 'x') if {'y','x'}.issubset(ds.dims) else ('lat', 'lon')

    # stack variables and compute a validity mask (any variable non-NaN)
    stacked = ds.stack(points=dims)

    n = 1000_000
    N = stacked.dims['points']
    rng = np.random.default_rng(42)
    idx = rng.choice(N, size=min(n, N), replace=False)
    sample = stacked.isel(points=idx)
    sample = sample.load()
    simple = sample.drop(['x','y','points'])

    # get variables as rows (long)
    vals = simple.to_dataframe()  # multi-index (points, variable) -> value
    vals = vals.reset_index()  # columns -> variables

    #drop cols band and spatial_ref and points

    vals = vals.drop(columns=['band', 'spatial_ref', 'points'])
    #drop row with any na
    vals = vals.dropna()
    vals
    #calc spearman correlation
    spearman_matrix = vals.corr(method='spearman')

    print(spearman_matrix)

    # ------------------------------------------------------------------
    # Figure 8: esri/hm/cpi hexbin pair matrix
    # ------------------------------------------------------------------

    cols = ['esri', 'hm', 'cpi']
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
        'esri': (0, 1),
        'cpi':  (0, 1),
        'hm':   (-0.1, 0.3)
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
            ax.set_ylabel('Count', fontsize=18)
            ax.tick_params(axis='both', labelsize=16)

    for i, row_var in enumerate(cols):
        ax = g.axes[i, 0]
        if ax is not None:
            ax.set_ylabel('Count' if i == 0 else row_var, fontsize=18)

    for j, col_var in enumerate(cols):
        ax = g.axes[len(cols) - 1, j]
        if ax is not None:
            ax.set_xlabel(col_var, fontsize=18)

    plot_axes = [ax for row in g.axes for ax in row if ax is not None]
    if hexbin_artists and plot_axes:
        cbar = g.fig.colorbar(hexbin_artists[0], ax=plot_axes, fraction=0.04, pad=0.03)
        cbar.set_label('Pixel count (log scale)', fontsize=18)
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

    variables_to_transform = ['esri', 'hm', 'cpi']

    for var in variables_to_transform:
        print(f"Transforming {var}...")
        ds_transformed[var] = transform_xarray_layer(
            ds[var], 
            a_param=0.8, 
            b_param=3
        )

    # Now ds_transformed contains the beta-distributed data 
    # with all original coordinates (lat, lon) preserved.

    ds = ds_transformed
    ds = ds.drop('band')
    rgb = create_ternary_alpha_array(ds, "esri", "hm", "cpi")

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
    ds = ds.drop_vars(['esri', 'hm', 'cpi'])

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


def figS5():
    """Figure S5: HM change 2020->2040 as 11 regional zooms (Fig 1/2 styling)."""
    OUT.mkdir(parents=True, exist_ok=True)
    OCEAN = config.OCEAN_HEX
    TARGET_PX = config.TARGET_PX_ZOOM
    cmap = mcolors.LinearSegmentedColormap.from_list("my_custom_coolwarm", config.COOLWARM_STOPS)
    vmin, vmax = config.CLIM
    cbar_label = "HM change 2040-2020"

    def load_hm():
        da = rxr.open_rasterio(config.PATHS['hm_diff'], chunks="auto").squeeze(drop=True)
        return da.where((da >= -1) & (da <= 1))

    def subset_region(da, bbox, margin=1.0):
        lon0, lon1, lat0, lat1 = bbox
        x_asc = float(da.x[-1]) > float(da.x[0])
        y_asc = float(da.y[-1]) > float(da.y[0])
        xs = slice(lon0 - margin, lon1 + margin) if x_asc else slice(lon1 + margin, lon0 - margin)
        ys = slice(lat0 - margin, lat1 + margin) if y_asc else slice(lat1 + margin, lat0 - margin)
        sub = da.sel(x=xs, y=ys)
        ny, nx = sub.sizes["y"], sub.sizes["x"]
        stride = max(1, int(np.ceil(max(ny, nx) / TARGET_PX)))
        return sub.isel(x=slice(None, None, stride), y=slice(None, None, stride)).compute()

    def render(name, slug, bbox, sub):
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(projection=ccrs.Robinson())
        ax.set_facecolor(OCEAN)
        mesh = ax.pcolormesh(
            sub.x.values, sub.y.values, sub.values,
            transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax,
            shading="auto", rasterized=True,
        )
        ax.set_extent(list(bbox), crs=ccrs.PlateCarree())
        ax.spines["geo"].set_edgecolor("black")
        ax.spines["geo"].set_linewidth(0.4)
        ax.set_title(name, fontsize=13)
        cbar = fig.colorbar(mesh, ax=ax, orientation="vertical",
                            shrink=0.55, pad=0.02, extend="both")
        cbar.set_label(cbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7, width=0.4, length=2)
        cbar.outline.set_linewidth(0.3)
        out = OUT / f"figS5_{slug}.png"
        fig.savefig(out, dpi=config.DPI_PLOT, bbox_inches="tight")
        plt.close(fig)
        return out

    da = load_hm()
    for slug, (name, bbox) in config.REGIONS.items():
        sub = subset_region(da, bbox)
        out = render(name, slug, bbox, sub)
        print(f"  saved {out.name}  ({sub.sizes['y']}x{sub.sizes['x']} cells)", flush=True)


def figS6():
    """Figure S6: 2040 natural-lands forecast classes as 11 regional zooms."""
    from matplotlib.patches import Patch
    OUT.mkdir(parents=True, exist_ok=True)
    OCEAN = config.OCEAN_HEX
    TARGET_PX = config.TARGET_PX_ZOOM
    class_colors = config.CLASS_COLORS
    class_labels = config.CLASS_LEGEND_LABELS
    cmap = mcolors.ListedColormap(class_colors)
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    def load_layers():
        raster = rxr.open_rasterio(config.PATHS['raster_classes'], chunks="auto").squeeze(drop=True)
        hm = rxr.open_rasterio(config.PATHS['hm_diff'], chunks="auto").squeeze(drop=True)
        return raster, hm

    def subset_region(raster, hm, bbox, margin=1.0):
        lon0, lon1, lat0, lat1 = bbox
        x_asc = float(raster.x[-1]) > float(raster.x[0])
        y_asc = float(raster.y[-1]) > float(raster.y[0])
        xs = slice(lon0 - margin, lon1 + margin) if x_asc else slice(lon1 + margin, lon0 - margin)
        ys = slice(lat0 - margin, lat1 + margin) if y_asc else slice(lat1 + margin, lat0 - margin)
        rsub = raster.sel(x=xs, y=ys)
        ny, nx = rsub.sizes["y"], rsub.sizes["x"]
        stride = max(1, int(np.ceil(max(ny, nx) / TARGET_PX)))
        rsub = rsub.isel(x=slice(None, None, stride), y=slice(None, None, stride)).compute()
        hsub = hm.sel(x=rsub.x, y=rsub.y, method="nearest").compute()
        vals = rsub.values.astype("float32")
        ocean = ~(hsub.values > -1e30)
        vals[ocean] = np.nan
        vals[(vals < 0) | (vals > 4)] = np.nan
        return rsub.x.values, rsub.y.values, vals

    def render(name, slug, bbox, x, y, vals):
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(projection=ccrs.Robinson())
        ax.set_facecolor(OCEAN)
        ax.pcolormesh(
            x, y, vals, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm,
            shading="auto", rasterized=True,
        )
        ax.set_extent(list(bbox), crs=ccrs.PlateCarree())
        ax.spines["geo"].set_edgecolor("black")
        ax.spines["geo"].set_linewidth(0.4)
        ax.set_title(name, fontsize=13)
        handles = [Patch(facecolor=c, edgecolor="0.3", linewidth=0.3, label=lab)
                   for c, lab in zip(class_colors, class_labels)]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  fontsize=7, frameon=False, handlelength=1.2, handleheight=1.2,
                  borderaxespad=0.0)
        out = OUT / f"figS6_{slug}.png"
        fig.savefig(out, dpi=config.DPI_PLOT, bbox_inches="tight")
        plt.close(fig)
        return out

    raster, hm = load_layers()
    for slug, (name, bbox) in config.REGIONS.items():
        x, y, vals = subset_region(raster, hm, bbox)
        out = render(name, slug, bbox, x, y, vals)
        print(f"  saved {out.name}  ({vals.shape[0]}x{vals.shape[1]} cells)", flush=True)


def fig10():
    """Figure 10: unprotected intact-lands loss 2020->2040 (maps + radials) + stats CSV.

    Also writes unprotected_loss_stats.csv, consumed by tables_s2_s9().
    """
    import gc
    import os
    import matplotlib.patches as mpatches
    import geopandas as gpd
    from rasterio.features import rasterize

    OUT.mkdir(parents=True, exist_ok=True)
    DATA = config.DATA_DIR
    PATH_HM_2020 = config.PATHS['hm_observed_2020']
    PATH_HM_CENTRAL = config.PATHS['hm_central_2040']
    PATH_HM_UPPER = config.PATHS['hm_upper_2040']
    PATH_PA = config.PATHS['hm_static_iucn_strict']
    PATH_ECO = DATA / 'Ecoregions2017' / 'Ecoregions2017.shp'

    INTACT_THRESHOLD = 0.1
    PROTECTED_VALUE = 1.0
    COARSEN = 4

    COLOR_PROTECTED = '#a8e47e'
    COLOR_REDBROWN = '#a28181'
    COLOR_LOST = '#e99060'
    COLOR_PERSISTENT = '#c0c0c0'
    COLOR_LAND_BASE = '#E0E0E0'
    COLOR_OCEAN = '#F4FCFF'

    def _load_raster(path: Path) -> xr.DataArray:
        da = rxr.open_rasterio(str(path), chunks=None).squeeze('band', drop=True)
        return da.where((da >= 0.0) & (da <= 1.0))


    def _summary(name: str, mask: np.ndarray) -> None:
        print(f"  {name}: {int(mask.sum()):,} pixels")


    def plot_loss_map(lost_mask: np.ndarray, land_mask: np.ndarray,
                      x_coords: np.ndarray, y_coords: np.ndarray,
                      out_path: Path) -> None:
        # Ternary encoding so we can coarsen the three states in one .max() pass:
        # 0 = ocean, 1 = land but not lost, 2 = lost. .max() preserves the highest-priority class.
        combined = land_mask.astype(np.uint8)
        combined[lost_mask] = 2
        da = xr.DataArray(combined, dims=('y', 'x'),
                          coords={'x': x_coords, 'y': y_coords})
        da = da.coarsen(x=COARSEN, y=COARSEN, boundary='trim').max()
        coarse = da.values
        x_c = da['x'].values
        y_c = da['y'].values

        # NaN where ocean so set_facecolor shows through; 0/1 for the cmap stops.
        display = np.where(coarse == 0, np.nan,
                           np.where(coarse == 2, 1.0, 0.0)).astype(np.float32)

        cmap = mcolors.LinearSegmentedColormap.from_list(
            'loss_cmap', [COLOR_LAND_BASE, COLOR_LOST]
        )
        cmap.set_bad(color=COLOR_OCEAN, alpha=0.0)  # ocean = transparent → ax facecolor shows

        fig = plt.figure(figsize=(13, 7))
        ax = plt.axes(projection=ccrs.Robinson())
        ax.set_global()
        ax.set_facecolor(COLOR_OCEAN)

        ax.pcolormesh(
            x_c, y_c, display,
            cmap=cmap, vmin=0.0, vmax=1.0,
            transform=ccrs.PlateCarree(),
            shading='nearest',
            rasterized=True,
        )

        ax.spines['geo'].set_visible(True)
        ax.spines['geo'].set_edgecolor('black')
        ax.spines['geo'].set_linewidth(0.3)

        fig.savefig(out_path, dpi=config.DPI_MAP, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  saved {out_path.name}")


    def plot_realm_radial(stats: pd.DataFrame, forecast: str, out_path: Path) -> None:
        pct_p_col = 'pct_protected'                       # static, no forecast suffix
        pct_rb_col = f'pct_red_brown_{forecast}'
        pct_l_col = f'pct_lost_{forecast}'

        # Drop Antarctica per request
        stats = stats[stats['REALM'] != 'Antarctica'].copy()

        realm_totals = stats.groupby('REALM')['eco_land_total'].sum().sort_values(ascending=False)
        realms = realm_totals.index.tolist()

        # Biomes present anywhere in the data (numbered key for the figure legend)
        biomes_in_data = (
            stats[['BIOME_NUM', 'BIOME_NAME']]
            .drop_duplicates()
            .sort_values('BIOME_NUM')
            .reset_index(drop=True)
        )

        nrows, ncols = 2, 4
        fig, axes = plt.subplots(nrows, ncols, figsize=(17, 10.5),
                                 subplot_kw=dict(projection='polar'),
                                 gridspec_kw=dict(hspace=0.05, wspace=0.30))
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
                    by=[pct_p_col, 'eco_land_total'], ascending=[False, False]
                )
                n = len(sub)
                thetas = cursor + arm_w * (np.arange(n) + 0.5)
                p = (sub[pct_p_col] / 100.0).values
                rb = (sub[pct_rb_col] / 100.0).values
                l = (sub[pct_l_col] / 100.0).values
                grey = np.clip(1.0 - p - rb - l, 0.0, 1.0)

                # Stack order (centre → rim): protected, grey, lost, red-brown
                ax.bar(thetas, p, width=arm_w * 0.95, bottom=r_base,
                       color=COLOR_PROTECTED, linewidth=0, align='center')
                ax.bar(thetas, grey, width=arm_w * 0.95, bottom=r_base + p,
                       color=COLOR_PERSISTENT, linewidth=0, align='center')
                ax.bar(thetas, l, width=arm_w * 0.95, bottom=r_base + p + grey,
                       color=COLOR_LOST, linewidth=0, align='center')
                ax.bar(thetas, rb, width=arm_w * 0.95, bottom=r_base + p + grey + l,
                       color=COLOR_REDBROWN, linewidth=0, align='center')

                # 30%-of-the-bar reference arc, contained within this biome group
                arc_r = r_base + 0.30
                arc_theta = np.linspace(cursor, cursor + arm_w * n, 64)
                ax.plot(arc_theta, np.full_like(arc_theta, arc_r),
                        color='black', alpha=0.5, linewidth=1.0, zorder=5)

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
            mpatches.Patch(color=COLOR_PROTECTED, label='Protected'),
            mpatches.Patch(color=COLOR_LOST, label='Natural lands loss 2040'),
            mpatches.Patch(color=COLOR_PERSISTENT, label='Still Natural 2040'),
            mpatches.Patch(color=COLOR_REDBROWN, label='Non-natural 2020'),
        ]
        fig.legend(handles=color_patches, loc='lower center', ncol=4,
                   frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.115))

        biome_handles = [
            mpatches.Patch(facecolor='none', edgecolor='none',
                           label=f"{int(b.BIOME_NUM)} — {b.BIOME_NAME}")
            for _, b in biomes_in_data.iterrows()
        ]
        fig.legend(handles=biome_handles, loc='lower center',
                   bbox_to_anchor=(0.5, 0.015),
                   ncol=3, fontsize=9, frameon=False,
                   handlelength=0, handletextpad=0,
                   title='Biome key', title_fontsize=10)

        fig.tight_layout(rect=[0.0, 0.20, 1.0, 0.98])
        fig.savefig(out_path, dpi=config.DPI_PLOT, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  saved {out_path.name}")


    def main() -> None:
        print("=== plot_unprotected_loss.py ===")
        OUT.mkdir(parents=True, exist_ok=True)

        # [1] Protected areas
        print("[1] Loading protected-area raster…")
        pa_da = _load_raster(PATH_PA)
        transform = pa_da.rio.transform()
        shape = pa_da.shape
        x_coords = pa_da['x'].values
        y_coords = pa_da['y'].values
        protected = (pa_da.values == PROTECTED_VALUE)
        del pa_da
        _summary("protected", protected)

        # [2] HM 2020
        print("[2] Loading HM 2020…")
        hm_2020 = _load_raster(PATH_HM_2020)
        hm_2020_vals = hm_2020.values
        land_mask = np.isfinite(hm_2020_vals)
        intact_2020 = (hm_2020_vals < INTACT_THRESHOLD) & land_mask
        del hm_2020, hm_2020_vals
        _summary("land", land_mask)
        _summary("intact_2020", intact_2020)

        protected_land = protected & land_mask                     # green numerator
        intact_unprotected = intact_2020 & ~protected
        unprot_nonintact_2020 = land_mask & ~protected & ~intact_2020  # used to derive red-brown
        _summary("protected_land", protected_land)
        _summary("intact_unprotected", intact_unprotected)
        _summary("unprot_nonintact_2020", unprot_nonintact_2020)
        del protected
        gc.collect()

        # [3] HM central
        print("[3] Loading HM central 2040…")
        hm_c = _load_raster(PATH_HM_CENTRAL)
        hm_c_vals = hm_c.values
        lost_central = intact_unprotected & (hm_c_vals >= INTACT_THRESHOLD)
        red_brown_central = unprot_nonintact_2020 & (hm_c_vals >= INTACT_THRESHOLD)
        del hm_c, hm_c_vals
        gc.collect()
        _summary("lost_central", lost_central)
        _summary("red_brown_central", red_brown_central)

        # [4] Map central
        skip_maps = os.environ.get('SKIP_MAPS') == '1'
        if not skip_maps:
            print("[4] Rendering map central…")
            plot_loss_map(lost_central, land_mask, x_coords, y_coords,
                          out_path=OUT / 'fig_unprotected_loss_map_central.png')
        else:
            print("[4] SKIP_MAPS=1, skipping map central")

        # [5] HM upper
        print("[5] Loading HM upper 2040…")
        hm_u = _load_raster(PATH_HM_UPPER)
        hm_u_vals = hm_u.values
        lost_upper = intact_unprotected & (hm_u_vals >= INTACT_THRESHOLD)
        red_brown_upper = unprot_nonintact_2020 & (hm_u_vals >= INTACT_THRESHOLD)
        del hm_u, hm_u_vals, unprot_nonintact_2020
        gc.collect()
        _summary("lost_upper", lost_upper)
        _summary("red_brown_upper", red_brown_upper)
        assert lost_upper.sum() >= lost_central.sum(), \
            "Sanity check failed: lost_upper should be >= lost_central"
        assert red_brown_upper.sum() >= red_brown_central.sum(), \
            "Sanity check failed: red_brown_upper should be >= red_brown_central"

        # [6] Map upper
        if not skip_maps:
            print("[6] Rendering map upper…")
            plot_loss_map(lost_upper, land_mask, x_coords, y_coords,
                          out_path=OUT / 'fig_unprotected_loss_map_upper.png')
        else:
            print("[6] SKIP_MAPS=1, skipping map upper")
        # land_mask is kept — it's the denominator (total ecoregion area) in zonal stats.
        gc.collect()

        # [7] Rasterize ecoregions
        print("[7] Rasterizing ecoregions…")
        eco = gpd.read_file(str(PATH_ECO), encoding='latin1')
        eco = eco[eco['REALM'].notna() & (eco['REALM'].astype(str) != 'N/A')].reset_index(drop=True)
        eco['idx'] = np.arange(1, len(eco) + 1, dtype=np.uint16)
        print(f"  kept {len(eco)} ecoregions across {eco['REALM'].nunique()} realms")
        shapes_iter = ((g, int(i)) for g, i in zip(eco.geometry, eco['idx']))
        eco_id = rasterize(shapes_iter, out_shape=shape, transform=transform,
                           fill=0, dtype='uint16', all_touched=False)
        flat_id = eco_id.ravel()
        N = len(eco)
        del eco_id
        gc.collect()

        # [8] Zonal stats
        print("[8] Computing zonal stats…")

        def zonal_sum(mask: np.ndarray) -> np.ndarray:
            return np.bincount(flat_id, weights=mask.ravel().astype(np.float32),
                               minlength=N + 1)[1:]

        # Forecast-independent "unprotected & non-natural in 2020" mask (used for the
        # table export's single 'Non-natural 2020' column).
        unprot_nonnatural_2020 = land_mask & ~protected_land & ~intact_2020

        stats = pd.DataFrame({
            'idx': eco['idx'].values,
            'REALM': eco['REALM'].values,
            'ECO_NAME': eco['ECO_NAME'].values,
            'BIOME_NUM': eco['BIOME_NUM'].astype(int).values,
            'BIOME_NAME': eco['BIOME_NAME'].values,
            'eco_land_total': zonal_sum(land_mask),
            'protected_land': zonal_sum(protected_land),
            'intact_total': zonal_sum(intact_2020),  # kept for reference
            'unprot_nonnatural_2020': zonal_sum(unprot_nonnatural_2020),
            'red_brown_central': zonal_sum(red_brown_central),
            'lost_central': zonal_sum(lost_central),
            'red_brown_upper': zonal_sum(red_brown_upper),
            'lost_upper': zonal_sum(lost_upper),
        })
        del (flat_id, intact_2020, protected_land, intact_unprotected,
             red_brown_central, lost_central, red_brown_upper, lost_upper,
             unprot_nonnatural_2020, land_mask)
        gc.collect()

        # Drop non-terrestrial biomes (BIOME_NUM 98=Lake, 99=Rock & Ice, etc.)
        stats = stats[stats['BIOME_NUM'].between(1, 14)].copy()
        stats = stats[stats['eco_land_total'] > 0].copy()
        print(f"  {len(stats)} ecoregions across "
              f"{stats['REALM'].nunique()} realms and "
              f"{stats['BIOME_NUM'].nunique()} biomes")

        # All percentages are of total ecoregion land area.
        stats['pct_protected'] = 100.0 * stats['protected_land'] / stats['eco_land_total']
        stats['pct_unprot_nonnatural_2020'] = (
            100.0 * stats['unprot_nonnatural_2020'] / stats['eco_land_total']
        )
        for f in ('central', 'upper'):
            stats[f'pct_red_brown_{f}'] = 100.0 * stats[f'red_brown_{f}'] / stats['eco_land_total']
            stats[f'pct_lost_{f}'] = 100.0 * stats[f'lost_{f}'] / stats['eco_land_total']
            # grey = everything else (still-intact + recovery + tiny HM-2040 NaN gap)
            stats[f'pct_grey_{f}'] = (100.0 - stats['pct_protected']
                                      - stats[f'pct_red_brown_{f}']
                                      - stats[f'pct_lost_{f}'])

        for f in ('central', 'upper'):
            s = (stats['pct_protected'] + stats[f'pct_red_brown_{f}']
                 + stats[f'pct_lost_{f}'] + stats[f'pct_grey_{f}'])
            assert np.allclose(s, 100.0, atol=0.01), f"closure failed for {f}"

        stats.to_csv(OUT / 'unprotected_loss_stats.csv', index=False)
        print(f"  saved unprotected_loss_stats.csv")

        # [9-10] Radial plots
        print("[9] Rendering radial central…")
        plot_realm_radial(stats, 'central', OUT / 'fig_unprotected_loss_radial_central.png')
        print("[10] Rendering radial upper…")
        plot_realm_radial(stats, 'upper', OUT / 'fig_unprotected_loss_radial_upper.png')

        print("=== done ===")

    main()
