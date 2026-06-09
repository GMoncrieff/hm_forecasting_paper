"""Shared, output-neutral helpers for the paper figures.

Pure-logic helpers (ternary colour scheme, coordinate formatting, layer
transform) live here; the heavier matplotlib/cartopy map builders are appended
in a later section. All values come from ``config`` so the figures stay
byte-identical to the original standalone scripts.
"""
import io
import urllib.request

import numpy as np
import xarray as xr
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import beta

import config

# Module-level numpy views of the ternary constants (the original notebook used
# np.array defaults; reproducing them keeps the maths identical).
_TERNARY_COLORS = np.asarray(config.TERNARY_COLORS, dtype=np.float32)
_BACKGROUND_RGB = np.asarray(config.BACKGROUND_RGB, dtype=np.float32)
_OCEAN_RGB = np.asarray(config.OCEAN_RGB, dtype=np.float32)


def fmt_coord(lat, lon):
    lat_s = f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}"
    lon_s = f"{abs(lon):.2f}°{'E' if lon >= 0 else 'W'}"
    return lat_s, lon_s


def transform_xarray_layer(da, a_param=0.8, b_param=3, random_state=42):
    """Applies Quantile Transform -> Beta Transform to a single xarray DataArray.

    Handles NaNs automatically (masks them out during transformation).
    """
    values_flat = da.values.flatten()
    valid_mask = ~np.isnan(values_flat)

    if not valid_mask.any():
        return da

    valid_data = values_flat[valid_mask].reshape(-1, 1)

    qt = QuantileTransformer(output_distribution='uniform',
                             random_state=random_state,
                             n_quantiles=min(len(valid_data), 1000))
    uniform_data = qt.fit_transform(valid_data)

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
                               sample_per_block=200_000):
    """xarray wrapper for ternary_alpha_rgb, two-pass and processed in row-blocks."""
    def _da(name):
        da = ds[name]
        return da.squeeze('band') if 'band' in da.dims else da

    d1, d2, d3 = _da(v1_var), _da(v2_var), _da(v3_var)
    H, W = d1.shape

    pts = _alpha_percentile_points(transparent_q, alpha_levels)
    samples = []
    for y0 in range(0, H, block_rows):
        sl = slice(y0, min(y0 + block_rows, H))
        mb = np.maximum(np.maximum(d1.isel(y=sl).values, d2.isel(y=sl).values),
                        d3.isel(y=sl).values)
        v = mb[np.isfinite(mb)]
        if v.size:
            step = max(1, v.size // sample_per_block)
            samples.append(v[::step])
    allv = np.concatenate(samples) if samples else np.array([0.0], dtype=np.float32)
    thr = np.percentile(allv, pts).astype(np.float32)
    print(f"  opacity thresholds @ pct {np.round(pts,2).tolist()} -> "
          f"max-values {np.round(thr,4).tolist()}  (n_sample={allv.size:,})", flush=True)

    out = np.empty((H, W, 3), dtype=np.float32)
    for y0 in range(0, H, block_rows):
        sl = slice(y0, min(y0 + block_rows, H))
        out[sl] = ternary_alpha_rgb(
            d1.isel(y=sl).values, d2.isel(y=sl).values, d3.isel(y=sl).values,
            alpha_thresholds=thr, alpha_levels=alpha_levels, base_alpha=base_alpha,
            background=background, invalid_color=invalid_color)
    return out
