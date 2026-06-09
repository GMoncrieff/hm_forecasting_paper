"""Figure-generating functions. Each writes its outputs to config.OUTPUT_DIR.

Bodies are ported from the standalone plot_*.py scripts; the shared global-map /
inset scaffolding (Figs 2, S1-S4) is delegated to utils, while Fig 3's
categorical map and Figs 8/9's RGB map keep their own verbatim bodies.
"""
import numpy as np
import xarray as xr
import rioxarray as rxr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap
import hvplot.xarray  # noqa: F401  (registers .hvplot accessor)
import holoviews as hv

import config
import utils

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
