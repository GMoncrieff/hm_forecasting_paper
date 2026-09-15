"""Shared, output-neutral helpers for the paper figures.

Pure-logic helpers (ternary colour scheme, coordinate formatting, layer
transform) live here; the heavier matplotlib/cartopy map builders are appended
in a later section. All values come from ``config`` so the figures stay
byte-identical to the original standalone scripts.
"""
import io
import math
import urllib.request
from pathlib import Path

import numpy as np
import xarray as xr
import rioxarray as rxr
import rasterio
from rasterio.warp import transform_bounds
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import beta

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.path as mpath
from matplotlib.patches import Circle, Polygon
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import hvplot.xarray  # noqa: F401  (registers the .hvplot accessor)
import holoviews as hv

import config
import equal_area

# Module-level numpy views of the ternary constants (the original notebook used
# np.array defaults; reproducing them keeps the maths identical).
_TERNARY_COLORS = np.asarray(config.TERNARY_COLORS, dtype=np.float32)
_BACKGROUND_RGB = np.asarray(config.BACKGROUND_RGB, dtype=np.float32)
_OCEAN_RGB = np.asarray(config.OCEAN_RGB, dtype=np.float32)


def fmt_coord(lat, lon):
    lat_s = f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}"
    lon_s = f"{abs(lon):.2f}°{'E' if lon >= 0 else 'W'}"
    return lat_s, lon_s


def transform_xarray_layer(da, a_param=0.8, b_param=3, random_state=42,
                           fit_values=None):
    """Applies Quantile Transform -> Beta Transform to a single xarray DataArray.

    Handles NaNs automatically (masks them out during transformation).

    `fit_values` is a 1-D array the quantile map is *fitted* on, while the
    transform is applied to `da` itself. Passing an equal-area sample makes the
    colour scale reflect how much ground lies below a value rather than how
    many pixels, without moving the rendered map off its native grid.

    Two things this pins down that the bare sklearn call did not:
      * subsample - QuantileTransformer defaults to subsample=10_000, so the
        map was previously fitted on 10,000 pixels out of ~150 million. The
        fit basis is now explicit and the whole of it is used.
      * n_quantiles is taken from the fit basis, not from the applied array.
    """
    values_flat = da.values.flatten()
    valid_mask = ~np.isnan(values_flat)

    if not valid_mask.any():
        return da

    valid_data = values_flat[valid_mask].reshape(-1, 1)

    basis = valid_data if fit_values is None else np.asarray(
        fit_values, dtype=np.float64).ravel()[:, None]
    basis = basis[np.isfinite(basis[:, 0])]
    if basis.size == 0:
        basis = valid_data

    qt = QuantileTransformer(output_distribution='uniform',
                             random_state=random_state,
                             n_quantiles=min(len(basis), 1000),
                             subsample=len(basis))
    qt.fit(basis)
    uniform_data = qt.transform(valid_data)

    transformed_valid = beta.ppf(uniform_data, a_param, b_param)

    new_values_flat = np.full(values_flat.shape, np.nan, dtype=np.float32)
    new_values_flat[valid_mask] = transformed_valid.flatten()

    return da.copy(data=new_values_flat.reshape(da.shape))


def _alpha_percentile_points(transparent_q=config.TRANSPARENT_Q,
                             alpha_levels=config.ALPHA_LEVELS):
    """Percentile cut points that split the top (1 - transparent_q) into equal-
    count bins. e.g. transparent_q=0.75, 4 levels -> [75, 81.25, 87.5, 93.75]."""
    return np.linspace(transparent_q * 100.0, 100.0, len(alpha_levels) + 1)[:-1]


def ternary_alpha_rgb(v1, v2, v3,
                      n_bins=config.TERNARY_N_BINS,
                      alpha_thresholds=None,
                      alpha_levels=config.ALPHA_LEVELS,
                      base_alpha=config.BASE_ALPHA,
                      transparent_q=config.TRANSPARENT_Q,
                      colors=_TERNARY_COLORS,
                      background=_BACKGROUND_RGB,
                      invalid_color=_OCEAN_RGB):
    """Three layers -> categorical ternary RGB (..., 3) with magnitude as opacity."""
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)
    v3 = np.asarray(v3, dtype=np.float32)

    invalid = ~(np.isfinite(v1) & np.isfinite(v2) & np.isfinite(v3))
    total = v1 + v2 + v3
    safe = np.where(total > 0, total, np.float32(1.0))   # avoid /0

    eps = np.float32(1e-6)

    def _idx(v):
        p = v / safe
        p = np.where(total > 0, p, np.float32(1.0 / 3.0))
        p = p * (1 - 3 * eps) + eps                       # nudge off lattice/edges
        return np.floor(p * n_bins)

    i1, i2, i3 = _idx(v1), _idx(v2), _idx(v3)
    add = np.where(np.rint(n_bins - (i1 + i2 + i3)) == 1,
                   np.float32(1.0), np.float32(2.0))       # 1 = up-triangle, 2 = down
    inv3n = np.float32(1.0 / (3 * n_bins))
    c1 = (3 * i1 + add) * inv3n
    c2 = (3 * i2 + add) * inv3n
    c3 = (3 * i3 + add) * inv3n

    hue_r = c1 * colors[0, 0] + c2 * colors[1, 0] + c3 * colors[2, 0]
    hue_g = c1 * colors[0, 1] + c2 * colors[1, 1] + c3 * colors[2, 1]
    hue_b = c1 * colors[0, 2] + c2 * colors[1, 2] + c3 * colors[2, 2]

    maxv = np.maximum(np.maximum(v1, v2), v3)
    if alpha_thresholds is None:
        mv = maxv[np.isfinite(maxv)]
        pts = _alpha_percentile_points(transparent_q, alpha_levels)
        alpha_thresholds = (np.percentile(mv, pts) if mv.size
                            else np.full(len(alpha_levels), np.inf))
    thr = np.asarray(alpha_thresholds, dtype=np.float32)
    lut = np.array((base_alpha,) + tuple(alpha_levels), dtype=np.float32)  # len = levels+1
    A = lut[np.digitize(maxv, thr)]                        # 0 -> base_alpha, then rising

    one_minus_A = np.float32(1.0) - A
    r = A * hue_r + one_minus_A * background[0]
    g = A * hue_g + one_minus_A * background[1]
    b = A * hue_b + one_minus_A * background[2]

    rgb = np.clip(np.stack([r, g, b], axis=-1), 0, 1)
    rgb[invalid] = invalid_color
    return rgb


def create_ternary_alpha_array(ds, v1_var='esri', v2_var='hm', v3_var='cpi',
                               background=_BACKGROUND_RGB, invalid_color=_OCEAN_RGB,
                               transparent_q=config.TRANSPARENT_Q,
                               alpha_levels=config.ALPHA_LEVELS,
                               base_alpha=config.BASE_ALPHA, block_rows=512,
                               sample_per_block=200_000, fit_values=None):
    """xarray wrapper for ternary_alpha_rgb, two-pass and processed in row-blocks.

    `fit_values` is a dict of {var: 1-D array} drawn from the equal-area grid.
    When given, the opacity cut-points are percentiles of *ground area* and the
    map is merely rendered on the native grid. Without it the cut-points fall
    back to the native-grid stride below, which is not area-fair: capping every
    512-row band at `sample_per_block` equalises latitude bands rather than
    land, discarding both the cos(lat) weighting and the land fraction.
    """
    def _da(name):
        da = ds[name]
        return da.squeeze('band') if 'band' in da.dims else da

    d1, d2, d3 = _da(v1_var), _da(v2_var), _da(v3_var)
    H, W = d1.shape

    pts = _alpha_percentile_points(transparent_q, alpha_levels)
    if fit_values is not None:
        stackedv = np.maximum(np.maximum(fit_values[v1_var], fit_values[v2_var]),
                              fit_values[v3_var])
        allv = stackedv[np.isfinite(stackedv)]
        basis = 'equal-area sample'
    else:
        samples = []
        for y0 in range(0, H, block_rows):
            sl = slice(y0, min(y0 + block_rows, H))
            mb = np.maximum(np.maximum(d1.isel(y=sl).values, d2.isel(y=sl).values),
                            d3.isel(y=sl).values)
            v = mb[np.isfinite(mb)]
            if v.size:
                step = max(1, v.size // sample_per_block)
                samples.append(v[::step])
        allv = (np.concatenate(samples) if samples
                else np.array([0.0], dtype=np.float32))
        basis = 'native-grid stride (NOT area-fair)'
    if allv.size == 0:
        allv = np.array([0.0], dtype=np.float32)
    thr = np.percentile(allv, pts).astype(np.float32)
    print(f"  opacity thresholds @ pct {np.round(pts,2).tolist()} -> "
          f"max-values {np.round(thr,4).tolist()}  "
          f"(n_sample={allv.size:,}, {basis})", flush=True)

    out = np.empty((H, W, 3), dtype=np.float32)
    for y0 in range(0, H, block_rows):
        sl = slice(y0, min(y0 + block_rows, H))
        out[sl] = ternary_alpha_rgb(
            d1.isel(y=sl).values, d2.isel(y=sl).values, d3.isel(y=sl).values,
            alpha_thresholds=thr, alpha_levels=alpha_levels, base_alpha=base_alpha,
            background=background, invalid_color=invalid_color)
    return out


# ---------------------------------------------------------------------------
# Colormaps, dask, satellite fetch, and shared global-map / inset builders
# ---------------------------------------------------------------------------
def register_coolwarm_cmap():
    """Register the 'my_custom_coolwarm' diverging colormap (idempotent)."""
    custom_cmap_obj = mcolors.LinearSegmentedColormap.from_list(
        'manual_turbo', config.COOLWARM_STOPS)
    try:
        plt.colormaps.register(name='my_custom_coolwarm', cmap=custom_cmap_obj)
    except ValueError:
        plt.colormaps.unregister('my_custom_coolwarm')
        plt.colormaps.register(name='my_custom_coolwarm', cmap=custom_cmap_obj)


def exceedance_cmap_norm():
    """(cmap, norm) for the exceedance-probability maps (Figs 3, 6, S6).

    A BoundaryNorm on uneven breaks, not a linear Normalize. Most land has a
    small probability of crossing a threshold, so a linear scale puts nearly
    every pixel in the first colour and the 0.01-0.2 range - where the signal
    actually lives - collapses to one shade.
    """
    cmap = mcolors.ListedColormap(config.P_EXCEED_COLORS)
    # Transparent, NOT ocean. pcolormesh RENDERS nodata cells with the bad
    # colour rather than skipping them, so an opaque bad value is a layer drawn
    # on top of everything beneath it. Here NaN means "already above the
    # threshold in the base year", and painting that ocean-blue covered the
    # out-of-base land underlay completely - zero beige pixels survived in a
    # 6008x2840 render. Leaving it transparent lets the z-order do the work:
    # axes facecolor is ocean, the underlay is out-of-base land, the mesh is
    # probability. Callers that draw no underlay (Fig 6's imshow panels) set
    # their own bad colour.
    cmap.set_bad(alpha=0.0)
    norm = mcolors.BoundaryNorm(config.P_EXCEED_LEVELS, cmap.N)
    return cmap, norm


def register_class_cmap():
    """Register the 5-class 'raster_classes' colormap (idempotent)."""
    custom_cmap = ListedColormap(config.CLASS_COLORS)
    try:
        plt.colormaps.register(name='raster_classes', cmap=custom_cmap)
    except ValueError:
        plt.colormaps.unregister('raster_classes')
        plt.colormaps.register(name='raster_classes', cmap=custom_cmap)


def make_dask_client():
    """Start a Dask distributed client (used by the heavier map figures)."""
    from dask.distributed import Client
    return Client()


def fetch_esri_satellite(extent, size=256):
    """Fetch a square satellite image from ESRI World Imagery REST API."""
    xmin, xmax, ymin, ymax = extent
    url = (f"https://server.arcgisonline.com/ArcGIS/rest/services/"
           f"World_Imagery/MapServer/export?"
           f"bbox={xmin},{ymin},{xmax},{ymax}&bboxSR=4326&imageSR=4326"
           f"&size={size},{size}&format=png&f=image")
    with urllib.request.urlopen(url) as resp:
        img = plt.imread(io.BytesIO(resp.read()), format='png')
    return img[:, :, :3]  # drop alpha channel


def build_global_robinson_map(pds, *, cmap, clim, cbar_label, hide_geo_spine=False):
    """Global Robinson quadmesh map with the Figure 1/2 styling.

    Identical to the inline code shared by Figs 2 and S1-S4; the only varying
    pieces (colormap, clim, colorbar label, initial geo-spine visibility) are
    arguments. Returns (fig, ax). Insets are added separately via
    ``add_circular_insets``.
    """
    hv.extension('matplotlib')
    plot = pds.hvplot.quadmesh(
        x='x', y='y',
        frame_width=1000,
        frame_height=700,
        pixel_ratio=6,
        xlabel='Longitude',
        ylabel='Latitude',
        rasterize=True,
        projection=ccrs.Robinson(),
        global_extent=True,
        cmap=cmap,
        ).opts(
        clim=clim
    )
    fig = hv.render(plot, backend='matplotlib')
    ax = fig.axes[0]

    ax.set_facecolor(config.OCEAN_HEX)
    ax.spines['geo'].set_visible(not hide_geo_spine)
    ax.spines['geo'].set_edgecolor('black')
    ax.spines['geo'].set_linewidth(0.3)
    ax.set_title('')

    for a in fig.axes[1:]:
        pos = a.get_position()
        a.set_position([pos.x0, pos.y0 + pos.height * 0.25, pos.width * 0.4, pos.height * 0.3])
        a.tick_params(labelsize=3, width=0.3, length=2)
        for spine in a.spines.values():
            spine.set_linewidth(0.3)
        a.set_ylabel(cbar_label, fontsize=4)

    return fig, ax


def add_circular_insets(fig, ax, pds, *, cmap, vmin=None, vmax=None, norm=None,
                        under=None):
    """Add the 3 circular zoom insets used by Figs 2, 3 and S1-S4.

    ``cmap`` is a registered colormap name or a Colormap; NaNs render as ocean.
    ``pds`` is the same DataArray plotted in the main map.

    Pass ``norm`` instead of ``vmin``/``vmax`` for a non-linear scale - the
    probability maps use a BoundaryNorm, which has no meaningful vmin/vmax.
    ``under`` is an optional second DataArray drawn beneath the data as a flat
    colour, which is how Fig 3 shows land that is outside its base: three
    distinct states (ocean, out-of-base land, probability) instead of two.

    The inset is sized off the *main map axes*, in inches, not off the figure.
    config.INSET_SIZE used to be a figure fraction, which only produced the
    intended 0.34 in circle on the 4x4 in canvas hv.render() returns for Fig 2;
    on the hand-built 13.9 x 7.6 in canvas of Figs 3a/3b the same fraction gave
    a 1.18 x 0.65 in box - both far too large and elliptical, since the circular
    border is drawn in axes-fraction space and inherits the box's aspect.
    Measuring against the map instead reproduces Fig 2 exactly and makes every
    other map match it whatever its canvas.
    """
    robinson = ccrs.Robinson()
    platecarree = ccrs.PlateCarree()

    inset_cmap = (plt.get_cmap(cmap) if isinstance(cmap, str) else cmap).copy()
    if under is None:
        # No underlay, so nodata IS ocean and painting it is right (Figs 2,
        # S1-S4). With an underlay it is not: pcolormesh renders nodata with the
        # bad colour, which would cover the out-of-base land drawn beneath and
        # put the inset back to two states while the main map shows three.
        inset_cmap.set_bad(config.OCEAN_HEX)
    mesh_kw = {'norm': norm} if norm is not None else {'vmin': vmin, 'vmax': vmax}

    inset_defs = config.INSET_DEFS
    radius_deg = config.RADIUS_DEG

    # One diameter in inches, then back to a per-axis figure fraction, so the
    # box is square on the page (a circle, not an ellipse) on any canvas.
    #
    # Draw first: a GeoAxes keeps aspect with adjustable='box', so its position
    # is only final once it has been laid out. Measuring before that would size
    # the insets off a box matplotlib is about to shrink - and the anchor
    # transform below reads the same stale layout.
    fig.canvas.draw()
    fig_w_in, fig_h_in = fig.get_size_inches()
    map_w_in = ax.get_position().width * fig_w_in
    inset_in = config.INSET_MAP_FRAC * map_w_in
    w_frac, h_frac = inset_in / fig_w_in, inset_in / fig_h_in

    # Strokes are in POINTS, which is an absolute size - so the same linewidth
    # on a map three times wider draws three times thinner once both figures are
    # scaled to one page width. That is why Fig 3's locator rings and inset
    # borders came out as hairlines beside Fig 2's. Scaling every point-valued
    # size by the map's width against Fig 2's reproduces Fig 2 exactly (scale 1)
    # and matches it everywhere else.
    scale = map_w_in / config.INSET_REF_MAP_W_IN
    border_lw = config.INSET_BORDER_LW * scale
    spine_lw = config.INSET_SPINE_LW * scale
    marker_size = config.INSET_MARKER_SIZE * scale
    marker_lw = config.INSET_MARKER_LW * scale

    for ins in inset_defs:
        clat, clon = ins['center']
        alat, alon = ins['anchor']

        x_rob, y_rob = robinson.transform_point(alon, alat, platecarree)
        disp = ax.transData.transform([x_rob, y_rob])
        fx, fy = fig.transFigure.inverted().transform(disp)

        ax_ins = fig.add_axes(
            [fx - w_frac / 2, fy - h_frac / 2, w_frac, h_frac],
            projection=platecarree
        )
        ax_ins.set_facecolor(config.OCEAN_HEX)
        ax_ins.set_extent(
            [clon - radius_deg, clon + radius_deg, clat - radius_deg, clat + radius_deg],
            crs=platecarree
        )

        window = dict(
            x=slice(clon - radius_deg - 0.1, clon + radius_deg + 0.1),
            y=slice(clat + radius_deg + 0.1, clat - radius_deg - 0.1),
        )
        sub = pds.sel(**window)

        if under is not None:
            # Flat land colour first, so the probability layer's NaNs read as
            # "outside the base" rather than as ocean.
            u = under.sel(**window)
            ax_ins.pcolormesh(
                u.x.values, u.y.values, np.where(np.asarray(u.values), 1.0, np.nan),
                cmap=mcolors.ListedColormap([config.OUT_OF_BASE_HEX]),
                vmin=0.0, vmax=1.0, transform=platecarree, shading='auto')

        ax_ins.pcolormesh(
            sub.x.values, sub.y.values, sub.values,
            cmap=inset_cmap, transform=platecarree, shading='auto', **mesh_kw
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
                        facecolor='none', edgecolor='black',
                        linewidth=border_lw, zorder=6)
        ax_ins.add_patch(border)

        ax_ins.set_xticks([])
        ax_ins.set_yticks([])
        for spine in ax_ins.spines.values():
            spine.set_visible(False)

        ax.spines['geo'].set_visible(True)
        ax.spines['geo'].set_edgecolor('black')
        ax.spines['geo'].set_linewidth(spine_lw)

        x_loc, y_loc = robinson.transform_point(clon, clat, platecarree)
        ax.plot(x_loc, y_loc, 'o', color='black', markersize=marker_size,
                markerfacecolor='none', markeredgewidth=marker_lw,
                transform=ax.transData, zorder=10)


def plot_quantile_fan_legend(out_path, gamma=config.FAN_GAMMA,
                             alpha_min=config.FAN_ALPHA_MIN,
                             alpha_max=config.FAN_ALPHA_MAX,
                             color=config.FAN_COLOR,
                             intervals=(0.5, 0.8, 0.99),
                             p_range=None, orientation='vertical',
                             title=None, facecolor='white', n=512,
                             figsize=None, fontsize=None, bar_frac=None):
    """Standalone legend for the continuous quantile fan used by Figure 7.

    The alpha mapping below has to stay identical to the one the fan itself
    uses: alpha = alpha_min + (alpha_max - alpha_min) * w ** gamma, with
    w = 1 - 2|F - 0.5|. If the two drift apart the legend stops describing the
    figure, and nothing would flag it.

    ``bar_frac`` is the colour bar's thickness as a fraction of the figure's
    short side, and ``fontsize`` sets the title and the tick labels; both
    default to config. The bar used to be laid out by plt.subplots, so it filled
    the axes and came out as a wide slab with small type beside it. It carries
    no quantitative information across its thickness - only along it - so the
    thickness is wasted ink, and the labels are the part a reader actually uses.

    Ported from output/timeseries.py.
    """
    if fontsize is None:
        fontsize = config.FAN_LEGEND_FONTSIZE
    if bar_frac is None:
        bar_frac = config.FAN_LEGEND_BAR_FRAC
    lo, hi = p_range if p_range is not None else (0.0, 1.0)
    coord = np.linspace(lo, hi, n)
    w = 1.0 - 2.0 * np.abs(coord - 0.5)      # 1 at median, -> 0 in tails

    ticks, labels = [], []
    for c in sorted(intervals):
        for b in ((1.0 - c) / 2.0, (1.0 + c) / 2.0):
            if lo <= b <= hi:
                ticks.append(b)
                labels.append(f'{c:.0%}')
    if lo <= 0.5 <= hi:
        ticks.append(0.5)
        labels.append('median')
    if title is None:
        title = 'Prediction interval'

    a = alpha_min + (alpha_max - alpha_min) * np.clip(w, 0.0, 1.0) ** gamma

    if figsize is None:
        figsize = (1.9, 3.0) if orientation == 'vertical' else (4.6, 1.5)
    fig = plt.figure(figsize=figsize)

    # Place the bar explicitly rather than letting a subplot fill the canvas:
    # bar_frac is its thickness, and the rest of the short side is left to the
    # labels. tight_layout is not used with add_axes (it would undo this);
    # bbox_inches='tight' at save time trims whatever slack is left over.
    if orientation == 'vertical':
        rect = [0.04, 0.03, bar_frac, 0.87]
    else:
        rect = [0.04, 1.0 - bar_frac - 0.30, 0.90, bar_frac]
    ax = fig.add_axes(rect)

    rgb = mcolors.to_rgb(color)
    strip = np.zeros((n, 8, 4))
    strip[..., :3] = rgb
    strip[..., 3] = a[:, None]

    if orientation == 'vertical':
        img, extent = strip, [0.0, 1.0, lo, hi]
    else:
        img, extent = np.transpose(strip, (1, 0, 2)), [lo, hi, 0.0, 1.0]

    ax.set_facecolor(facecolor)
    ax.imshow(img, extent=extent, origin='lower', aspect='auto',
              interpolation='bilinear')

    tick_fs = fontsize
    if orientation == 'vertical':
        ax.set_ylim(lo, hi)
        ax.set_xticks([])
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.yaxis.tick_right()
        ax.tick_params(axis='y', length=3, labelsize=tick_fs, pad=3)
        ax.set_title(title, fontsize=fontsize, pad=8, loc='left')
    else:
        ax.set_xlim(lo, hi)
        ax.set_yticks([])
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.tick_params(axis='x', length=3, labelsize=tick_fs, pad=3)
        ax.set_xlabel(title, fontsize=fontsize)

    for side in ('left', 'right', 'top', 'bottom'):
        ax.spines[side].set_visible(False)

    ax.patch.set_edgecolor('0.7')
    ax.patch.set_linewidth(0.8)

    fig.savefig(str(out_path), dpi=config.DPI_PLOT, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_ternary_alpha_legend(out_path,
                              n_bins=config.TERNARY_N_BINS,
                              colors=_TERNARY_COLORS,
                              labels=config.TERNARY_LABELS,
                              background=_BACKGROUND_RGB,
                              alpha_levels=config.ALPHA_LEVELS,
                              base_alpha=config.BASE_ALPHA,
                              transparent_q=config.TRANSPARENT_Q):
    """Figure 9 colour key: full-opacity ternary hue triangle above an opacity bar."""
    h = np.sqrt(3) / 2.0

    def bary2xy(b):
        return (b[1] * 0.5 + b[2], b[1] * h)

    fig = plt.figure(figsize=(6.2, 7.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[6.0, 1.0], hspace=0.12)
    ax = fig.add_subplot(gs[0])
    axbar = fig.add_subplot(gs[1])

    for up in (True, False):
        tot = n_bins - 1 if up else n_bins - 2
        for i in range(tot + 1):
            for j in range(tot - i + 1):
                k = tot - i - j
                if up:
                    vb = [(i + 1, j, k), (i, j + 1, k), (i, j, k + 1)]
                    c = np.array([3 * i + 1, 3 * j + 1, 3 * k + 1], dtype=np.float32) / (3 * n_bins)
                else:
                    vb = [(i + 1, j + 1, k), (i + 1, j, k + 1), (i, j + 1, k + 1)]
                    c = np.array([3 * i + 2, 3 * j + 2, 3 * k + 2], dtype=np.float32) / (3 * n_bins)
                col = np.clip(c @ colors, 0, 1)
                verts = [bary2xy(np.array(v, dtype=float) / n_bins) for v in vb]
                ax.add_patch(Polygon(verts, closed=True, facecolor=tuple(col),
                                     edgecolor='white', linewidth=0.6))

    ax.add_patch(Polygon([(0, 0), (1, 0), (0.5, h)], closed=True,
                         edgecolor='black', facecolor='none', linewidth=2.0))
    lab = dict(fontsize=16, fontweight='bold')
    ax.text(-0.06, -0.05, labels[0], color=tuple(colors[0]), ha='right', va='top', **lab)
    ax.text(0.5, h + 0.06, labels[1], color=tuple(colors[1]), ha='center', va='bottom', **lab)
    ax.text(1.06, -0.05, labels[2], color=tuple(colors[2]), ha='left', va='top', **lab)
    # The sides used to carry 25/50/75 tick labels. The triangle encodes which
    # of the three layers dominates, not a readable mixing ratio, so the numbers
    # invited a precision the hue binning does not support.
    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(-0.18, h + 0.22)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Hue  =  which dataset dominates", fontsize=12, pad=2)

    n_lv = len(alpha_levels)
    q0 = transparent_q * 100.0
    width = (100.0 - q0) / n_lv
    Nx = 600
    xs = np.linspace(0, 100, Nx)
    A = np.full(Nx, base_alpha, dtype=float)
    for li, a in enumerate(alpha_levels):
        A[(xs >= q0 + li * width) & (xs <= q0 + (li + 1) * width)] = a
    ink = np.array([0.16, 0.16, 0.19])
    bar = A[:, None] * ink + (1 - A[:, None]) * background
    axbar.imshow(bar[None, :, :], extent=[0, 100, 0, 1], aspect='auto', origin='lower',
                 interpolation='nearest')
    for li in range(n_lv + 1):
        axbar.axvline(q0 + li * width, color='white', lw=0.8)
    axbar.axvline(q0, color='black', lw=1.6)
    axbar.add_patch(Polygon([(0, 0), (100, 0), (100, 1), (0, 1)], closed=True,
                            fill=False, edgecolor='black', lw=1.0))
    axbar.set_xlim(0, 100)
    axbar.set_ylim(0, 1)
    axbar.set_yticks([])
    axbar.set_xticks([0, int(q0), 100])
    axbar.set_xticklabels(['0', f'{int(q0)}', '100'])
    for sp in axbar.spines.values():
        sp.set_visible(False)
    axbar.set_xlabel(
        "Opacity  =  area-weighted magnitude percentile of max(ESRI, HM, CPI)\n"
        f"≤ {int(q0)}th pct of land area → transparent      ·      "
        f"top {int(100 - q0)}% → {n_lv} equal-area bins, rising to opaque",
        fontsize=9)

    fig.savefig(out_path, dpi=config.DPI_PLOT, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Equal-area analysis grid.
#
# The source rasters are EPSG:4326, where a cell's ground area falls off with
# cos(lat), so counting pixels on that grid over-weights high latitudes. Every
# statistic in the paper is therefore computed on a shared equal-area grid
# (config.EQUAL_AREA_CRS), onto which layers are warped on the fly. Maps are
# NOT warped - they render through cartopy, which handles area honestly, and
# leaving them on the native grid avoids re-plumbing every PlateCarree call.
# ---------------------------------------------------------------------------

# The grid machinery itself is a leaf module (numpy + rasterio only) so the
# stats.py console script can import it without pulling in matplotlib,
# cartopy and holoviews behind it. Re-exported here for the figure code.
AUTHALIC_RADIUS_M = equal_area.AUTHALIC_RADIUS_M
equal_area_grid = equal_area.equal_area_grid
open_equal_area = equal_area.open_equal_area
read_masked = equal_area.read_masked


def align_like(da, ref, *, tol=1e-6):
    """Snap `da` onto `ref`'s x/y coordinates when the two describe one grid.

    The probability COGs were written by a different tool than the HM rasters.
    Both declare the identical transform and shape, but the two toolchains
    build the coordinate arrays with slightly different floating-point
    arithmetic - a max difference of ~1e-11 degrees, roughly a micrometre.

    xarray aligns binary operations on coordinate *equality*, not proximity, so
    `p.where(hm < 0.1)` inner-joins 40,000 longitudes down to the 42 that match
    bit-for-bit and returns a 21x42 array. No warning is raised and the result
    is a map of 882 pixels, which is why this is a hard failure below rather
    than something to fix at each call site.

    Only the native-grid path needs this. Anything read through
    `open_equal_area` is warped onto one explicitly-constructed grid and is
    immune by construction.
    """
    if da.shape != ref.shape:
        raise ValueError(f'cannot align {da.shape} onto {ref.shape}: '
                         'different grids, not a floating-point difference')
    for dim in ('x', 'y'):
        drift = float(np.abs(da[dim].values - ref[dim].values).max())
        if drift > tol:
            raise ValueError(
                f'{dim} coordinates differ by {drift:g} > {tol:g}; these are '
                'genuinely different grids, so snapping them would be wrong')
    return da.assign_coords({'x': ref['x'], 'y': ref['y']})


def trim_wrapped_columns(x_coords, *arrays):
    """Drop the equal-area grid's antimeridian-straddling edge columns.

    `equal_area_grid` snaps the destination bounds outward to whole cells, so
    for EPSG:6933 the grid is ~470 m wider than the projection's valid domain
    at each side and the first and last columns straddle +/-180 degrees.
    Cartopy draws a quad that crosses the antimeridian as a polygon spanning
    the entire map, which puts a horizontal stripe across the figure at every
    latitude with land near the date line - most visibly around 72 N, where
    Chukotka and Wrangel Island sit.

    This is a display problem only. Those columns hold 748 of 131.4 M land
    cells (0.0006%), so trimming them here rather than reshaping the analysis
    grid keeps every reported number exactly as computed.

    Returns (x_coords, *arrays) with the offending columns removed from the
    last axis of each.
    """
    x = np.asarray(x_coords)
    if x.size < 3 or not config.EQUAL_AREA_CRS:
        return (x, *arrays)
    left, _, right, _ = transform_bounds(
        'EPSG:4326', config.EQUAL_AREA_CRS, -180.0, -1e-6, 180.0, 1e-6)
    limit = max(abs(left), abs(right))
    half = abs(float(x[1] - x[0])) / 2.0
    keep = np.abs(x) + half <= limit
    if keep.all():
        return (x, *arrays)
    return (x[keep], *[np.asarray(a)[..., keep] for a in arrays])


def open_equal_area_da(stack, path, grid, *, chunks=None, **kwargs):
    """`open_equal_area` as an xarray DataArray, for the rioxarray call sites.

    Always opens masked=True, so every nodata convention becomes NaN. Loading
    unmasked is what let 3.4e38 survive into arithmetic elsewhere in this repo:
    ocean minus ocean is exactly 0.0, which is finite, in range, and reads as a
    genuine "no change" observation.

    The returned array stays lazy when `chunks` is given; keep `stack` alive
    until it has been computed, or the underlying VRT closes beneath dask.
    """
    layer = open_equal_area(stack, path, grid, **kwargs)
    da = rxr.open_rasterio(layer, masked=True, chunks=chunks)
    return da.squeeze('band', drop=True) if 'band' in da.dims else da


# ---------------------------------------------------------------------------
# HM-change layers, computed live from the AA rasters (replaces the precomputed
# HM_DIFF.tif / hm_diff_obs.tif). masked=True so declared nodata -> NaN, which
# preserves the ocean mask through the subtraction (ocean stays NaN, not 0).
# ---------------------------------------------------------------------------
def load_hm_diff(chunks='auto'):
    """Expected HM change 2040-2020 = E[HM 2040] - observed 2020 (live).

    The blended "central" surface is E[Q] = the integral of Q(u) over [0,1] -
    the mean of the predictive distribution, not its median - so this is an
    expected change. Figures 2, S5 and 8/9 all read it through here.
    """
    central = rxr.open_rasterio(config.PATHS['hm_central_2040'], chunks=chunks, masked=True)
    obs2020 = rxr.open_rasterio(config.PATHS['hm_2020_aa'], chunks=chunks, masked=True)
    return central - obs2020


def load_hm_diff_bound(bound, chunks='auto'):
    """HM change 2040-2020 from a distribution *bound* rather than its mean.

    ``bound`` is 'lower' or 'upper': the 2.5th and 97.5th percentile of each
    pixel's own predictive distribution. Figures S5 and S6 show these beside
    Figure 2's expected change, so a reader can see the width of the forecast
    on the same colour scale.

    These are per-pixel percentiles, not scenarios. A map of the upper bound is
    the change realised only if every pixel lands on its unlucky outcome at the
    same time - the perfect-dependence extreme - which is why nothing in this
    repo sums or thresholds them. See the README.
    """
    if bound not in ('lower', 'upper'):
        raise ValueError(f"bound must be 'lower' or 'upper', got {bound!r}")
    b = rxr.open_rasterio(config.PATHS[f'hm_{bound}_2040'], chunks=chunks,
                          masked=True)
    obs2020 = rxr.open_rasterio(config.PATHS['hm_2020_aa'], chunks=chunks,
                                masked=True)
    return b - obs2020


def load_hm_diff_obs(chunks='auto'):
    """Observed HM change 2020-2000 = observed 2020 - observed 2000 (live)."""
    obs2020 = rxr.open_rasterio(config.PATHS['hm_2020_aa'], chunks=chunks, masked=True)
    obs2000 = rxr.open_rasterio(config.PATHS['hm_2000_aa'], chunks=chunks, masked=True)
    return obs2020 - obs2000
