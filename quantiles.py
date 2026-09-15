"""Reading the per-pixel quantile functions from the icechunk stores.

The distributional ConvLSTM emits a monotone rational-quadratic spline quantile
function per pixel, published as 64 levels in two icechunk repositories - one
for the 2020-based forecast (2025-2040), one for the 2000-based hindcast
(2005-2020). Everything in the paper that needs more than a summary surface -
Fig 5's calibration, Fig 7's fan and predictive density - reads through here.

Deliberately a leaf module, in the same spirit as equal_area: numpy + xarray +
icechunk only, no matplotlib behind it, so stats.py could import it without
dragging in the plotting stack.

The one trap this module exists to close: **the stores do not self-decode**.
`quantile_forecast` is int16 with attributes named `scale` and `sentinel`, not
the CF-standard `scale_factor` and `_FillValue`, so xarray hands back raw
integers. A value read without `decode()` is ~32,000x too large and every
threshold test silently passes. `decode()` reads both attributes off the array
rather than hard-coding them, because a re-export with a different scale would
otherwise be wrong in a way nothing would catch.
"""
import numpy as np
import xarray as xr

import config

QF_VAR = 'quantile_forecast'

# The store's chunks are (1, 64, 512, 512) - all quantile levels for a 512x512
# tile at one horizon in a single chunk. Reading on that lattice means one
# chunk per call instead of four; anything else multiplies the read.
CHUNK = 512


def open_qf(mode: str) -> xr.Dataset:
    """Open the 'forecast' or 'hindcast' quantile store, read-only.

    Values stay int16 here - call `decode` on whatever slice you take.
    """
    import icechunk

    key = {'forecast': 'qf_forecast', 'hindcast': 'qf_hindcast'}[mode]
    path = config.PATHS[key]
    if not path.exists():
        raise FileNotFoundError(f'{mode} quantile store not found: {path}')
    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(path)))
    return xr.open_zarr(repo.readonly_session('main').store, consolidated=False)


def decode(da: xr.DataArray) -> np.ndarray:
    """int16 -> float32 HM, with the sentinel collapsed to NaN.

    `scale` and `sentinel` are read from the array's own attributes. They are
    NOT the CF names xarray decodes automatically (`scale_factor`, `_FillValue`),
    which is exactly why this function has to exist.
    """
    scale = float(da.attrs['scale'])
    offset = float(da.attrs.get('offset', 0.0))
    sentinel = int(da.attrs['sentinel'])

    raw = np.asarray(da.values)
    out = raw.astype(np.float32) * np.float32(scale) + np.float32(offset)
    return np.where(raw == sentinel, np.float32(np.nan), out)


def levels(ds: xr.Dataset) -> np.ndarray:
    """The 64 probability levels, ascending."""
    return np.asarray(ds['quantile'].values, dtype=np.float64)


def pixel_qf(ds: xr.Dataset, lat: float, lon: float):
    """One pixel's quantile function at every horizon.

    Returns (p, v, actual_lat, actual_lon) with p of shape (n_q,) ascending and
    v of shape (n_time, n_q) - one row per horizon, values in HM units.
    """
    point = ds[QF_VAR].sel(latitude=lat, longitude=lon, method='nearest')
    point = point.sortby('quantile').compute()
    v = decode(point).astype(np.float64)
    if point.dims.index('quantile') != point.ndim - 1:
        v = np.moveaxis(v, point.dims.index('quantile'), -1)
    return (np.asarray(point['quantile'].values, dtype=np.float64), v,
            float(point['latitude']), float(point['longitude']))


def block_qf(ds: xr.Dataset, row: int, col: int, time_index: int,
             size: int = CHUNK):
    """One chunk-aligned tile of quantile functions.

    Returns v of shape (n_q, ny, nx) in HM units. `row`/`col` are the top-left
    pixel indices; align them to `CHUNK` or the read spans four chunks instead
    of one.
    """
    tile = ds[QF_VAR].isel(
        time=time_index,
        latitude=slice(row, row + size),
        longitude=slice(col, col + size),
    ).sortby('quantile').compute()
    return decode(tile).astype(np.float32)


# --- the quantile function as a distribution ---------------------------------
#
# Q is known at the levels p and treated as piecewise linear in u between them,
# constant outside. Two CDF values matter, and they differ wherever Q is flat:
#
#   F-(y) = measure{u : Q(u) <  y}
#   F+(y) = measure{u : Q(u) <= y}
#
# HM has a real atom at "no change", so plateaus are common and the gap between
# the two is not a rounding artefact. Ignoring it is what makes a PIT histogram
# spike at the ends and read as miscalibration that is not there.


def _bracket(p: np.ndarray, v: np.ndarray, y: np.ndarray):
    """(F-, F+) for value `y` under each column of `v`.

    `v` is (n_q, N) ascending down axis 0; `y` is (N,). Returns two (N,) arrays.
    """
    n_q = v.shape[0]
    lt = (v < y).sum(axis=0)          # index of the first level >= y
    le = (v <= y).sum(axis=0)         # index of the first level >  y

    f_minus = np.empty(y.shape, dtype=np.float64)
    f_plus = np.empty(y.shape, dtype=np.float64)

    # `below` must test le, not lt: when y equals the lowest quantile the value
    # is ON the support, not under it, and treating it as under collapses the
    # atom at that value to zero mass. HM has a real atom at no-change, so this
    # is the common case, not an edge case.
    below = le == 0                   # y strictly under the whole support
    above = lt >= n_q                 # y over the whole support
    flat = (le > lt) & ~below & ~above          # y sits on a plateau
    interp = (le == lt) & ~below & ~above       # y strictly inside a segment

    f_minus[below] = f_plus[below] = 0.0
    f_minus[above] = f_plus[above] = 1.0

    if flat.any():
        idx = np.nonzero(flat)[0]
        f_minus[idx] = p[lt[idx]]
        f_plus[idx] = p[le[idx] - 1]

    if interp.any():
        idx = np.nonzero(interp)[0]
        k = lt[idx]
        lo_v, hi_v = v[k - 1, idx], v[k, idx]
        lo_p, hi_p = p[k - 1], p[k]
        span = hi_v - lo_v
        # span == 0 is already handled by the `flat` branch; guard anyway so a
        # degenerate segment cannot produce a NaN that propagates silently.
        frac = np.where(span > 0, (y[idx] - lo_v) / np.where(span > 0, span, 1.0), 0.0)
        f_minus[idx] = f_plus[idx] = lo_p + (hi_p - lo_p) * frac

    return f_minus, f_plus


def cdf_at(p: np.ndarray, v: np.ndarray, y: np.ndarray) -> np.ndarray:
    """F+(y) - the ordinary CDF - for each column of `v`."""
    return _bracket(p, v, np.asarray(y, dtype=np.float64))[1]


def pit(p: np.ndarray, v: np.ndarray, y: np.ndarray, rng) -> np.ndarray:
    """Randomised probability integral transform.

    On a plateau the PIT is drawn uniformly between F- and F+. Without that
    randomisation the atom at no-change piles observations onto a handful of
    values and the histogram is non-uniform for a perfectly calibrated
    forecast.
    """
    f_minus, f_plus = _bracket(p, v, np.asarray(y, dtype=np.float64))
    u = rng.random(f_minus.shape)
    return f_minus + u * (f_plus - f_minus)


def bin_probs(p: np.ndarray, v: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """P(value in each bin) per column of `v`.

    `edges` is either (n_edges,) shared by every column, or (n_edges, N) when
    each column has its own edges - which is how Fig 5 turns a distribution
    over HM level into one over change, by shifting the edges by that pixel's
    2000 observation.

    Returns (n_edges - 1, N).
    """
    edges = np.asarray(edges, dtype=np.float64)
    n = v.shape[1]
    if edges.ndim == 1:
        edges = np.repeat(edges[:, None], n, axis=1)
    # One pass per edge: (n_q, N) per comparison instead of (n_q, n_edges, N).
    cdfs = np.stack([cdf_at(p, v, edges[j]) for j in range(edges.shape[0])])
    return np.diff(cdfs, axis=0)


def density(p: np.ndarray, v: np.ndarray, grid: np.ndarray,
            smooth: float = 0.0) -> np.ndarray:
    """Predictive density on `grid`, from one pixel's quantile function.

    f(Q(u)) = du/dQ, so the density is the gradient of the CDF. `smooth` is a
    Gaussian sigma in grid cells; the raw gradient of a 64-knot spline is
    steppy, and Fig 7 wants a line rather than a staircase.
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    # np.interp needs strictly increasing x; nudge plateaus apart by an amount
    # far below the int16 quantisation so the shape is untouched.
    v = np.maximum.accumulate(v)
    v = v + np.arange(v.size) * (1e-9 * max(np.ptp(v), 1.0))

    cdf = np.interp(grid, v, p, left=0.0, right=1.0)
    dens = np.gradient(cdf, grid)
    if smooth > 0:
        from scipy.ndimage import gaussian_filter1d
        dens = gaussian_filter1d(dens, smooth, mode='nearest')
    return np.clip(dens, 0.0, None)


# --- the continuous fan (ported from output/timeseries.py) -------------------


def anchor_to_observation(t, v, obs_t, obs_v, anchor_time=None):
    """Prepend the last observation so the fan tapers to a point.

    The store's first horizon is +5 years, so an unanchored fan starts already
    wide and reads as though the model were uncertain about a year it was
    given. Collapsing every quantile onto the observed value at `anchor_time`
    (default: the last observation strictly before the first forecast) puts the
    taper where it belongs.
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    obs_t = np.asarray(obs_t, dtype=float)
    obs_v = np.asarray(obs_v, dtype=float)

    if anchor_time is None:
        earlier = obs_t[(obs_t < t[0]) & np.isfinite(obs_v)]
        if earlier.size == 0:
            return t, v
        anchor_time = float(earlier.max())

    k = int(np.argmin(np.abs(obs_t - anchor_time)))
    if not np.isfinite(obs_v[k]):
        return t, v

    return (np.concatenate([[obs_t[k]], t]),
            np.column_stack([np.full(v.shape[0], obs_v[k]), v]))


def fan_image(t, p, v, ny=600, nt=600, mode='prob', gamma=1.0,
              normalize='global', pad_frac=0.03, ref_time=None):
    """Continuous opacity field for a quantile fan.

    t : (n_time,) numeric times;  p : (n_q,) ascending levels;
    v : (n_q, n_time) quantile values.  Returns (t_f, y, A, M) with A the
    opacity weight in [0, 1] and M True inside the fan's support.

    Ported from output/timeseries.py.
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    v = np.asarray(v, dtype=float)

    # resample in time so the image is smooth along x
    t_f = np.linspace(t[0], t[-1], nt)
    v = np.vstack([np.interp(t_f, t, row) for row in v])

    # np.interp needs a strictly increasing x, so enforce it
    v = np.maximum.accumulate(v, axis=0)
    eps = 1e-9 * max(np.ptp(v), 1.0)
    v = v + np.arange(v.shape[0])[:, None] * eps

    pad = pad_frac * np.ptp(v)
    y = np.linspace(v.min() - pad, v.max() + pad, ny)

    A = np.zeros((ny, nt))
    M = np.zeros((ny, nt), dtype=bool)

    for j in range(nt):
        # invert the quantile function: CDF evaluated on the y grid
        F = np.interp(y, v[:, j], p, left=0.0, right=1.0)
        inside = (y >= v[0, j]) & (y <= v[-1, j])

        if mode == 'prob':
            col = 1.0 - 2.0 * np.abs(F - 0.5)     # 1 at median, -> 0 in tails
        else:
            col = np.gradient(F, y)               # predictive density

        A[:, j] = np.where(inside, np.maximum(col, 0.0), 0.0)
        M[:, j] = inside

    # normalize off the real forecast, not the artificial taper
    if ref_time is None:
        ref_cols = np.ones(nt, dtype=bool)
    else:
        ref_cols = t_f >= float(ref_time)
        if not ref_cols.any():
            ref_cols = np.ones(nt, dtype=bool)

    if mode == 'density' and normalize == 'column':
        ref = np.maximum(A.max(axis=0, keepdims=True), 1e-12)
    else:
        ref = max(float(A[:, ref_cols].max()), 1e-12)

    A = np.clip(A / ref, 0.0, 1.0)

    return t_f, y, A ** gamma, M
