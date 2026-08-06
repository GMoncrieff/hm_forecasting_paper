"""The shared equal-area analysis grid.

The source rasters are EPSG:4326, where a cell's ground area falls off with
cos(lat) - about 1.00 km2 at the equator against 0.10 km2 at 84 N. Counting
pixels on that grid is therefore not counting area, and over-weights high
latitudes. Every quantity the paper reports is computed after warping onto the
grid built here, where one cell is one constant patch of ground.

Deliberately a leaf module: numpy + rasterio only, no config-independent state
beyond what it reads at call time. stats.py is a console script and importing
utils would drag in matplotlib, cartopy, sklearn and holoviews behind it - a 20x
import cost and a headless-render failure mode for something that only needs a
VRT. utils re-exports these names for the figure code.
"""
import math

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

import config

AUTHALIC_RADIUS_M = 6371007.181   # WGS 84 authalic sphere, as equal-area CRSs use


def equal_area_grid(ref):
    """(transform, width, height) of the shared grid, or None when disabled.

    Derived once from a reference raster and reused for every warp, so all
    layers land on an identical grid. With nearest resampling that means a
    given destination cell draws from the same source cell in every raster, and
    differencing two warped layers stays pixel-exact.
    """
    if not config.EQUAL_AREA_CRS:
        return None
    res = config.EQUAL_AREA_RES
    left, bottom, right, top = transform_bounds(
        ref.crs, config.EQUAL_AREA_CRS, *ref.bounds)
    # Snap outward to whole cells so the grid origin is a round number.
    left, bottom = math.floor(left / res) * res, math.floor(bottom / res) * res
    right, top = math.ceil(right / res) * res, math.ceil(top / res) * res
    return (Affine(res, 0, left, 0, -res, top),
            int(round((right - left) / res)), int(round((top - bottom) / res)))


def open_equal_area(stack, path, grid, *, src_nodata=None, fill=None):
    """Open `path` warped onto `grid`, registering it on `stack` (an ExitStack).

    Returns the raw dataset unchanged when `grid` is None (equal-area counting
    disabled), so callers need no branch of their own.

    Two traps this guards, both observed in this repo's own data:

      * The inputs disagree about nodata - HM_observed_* declare +3.4e38,
        hm_static_iucn_strict/hm_diff* declare -3.4e38, ESRI/CPI declare NaN,
        and split_mask/raster_classes declare nothing at all. Pass `src_nodata`
        to supply the value when the file omits or misstates it.
      * NaN cannot be represented in an integer raster, so requesting a NaN
        fill on uint8 makes GDAL promote the whole VRT to float64 - a silent
        4x memory blow-up on a 493 M-cell grid. Integer rasters must name an
        in-dtype sentinel `fill` instead.

    Note the destination grid is snapped outward by under one cell row, so for
    EPSG:6933 there is no meaningful out-of-footprint region; `fill` matters for
    dtype and for CRSs whose domain does not fill their bounding rectangle.
    """
    src = stack.enter_context(rasterio.open(path))
    if grid is None:
        return src
    transform, width, height = grid

    is_float = np.issubdtype(np.dtype(src.dtypes[0]), np.floating)
    if fill is None:
        if not is_float:
            raise ValueError(
                f'{getattr(path, "name", path)} is {src.dtypes[0]}; pass '
                'fill=<sentinel>. A NaN fill is silently dropped for integer '
                'rasters, and the polar gap then reads as a real class value.')
        fill = float('nan')

    # Passing src_nodata=None EXPLICITLY is not the same as omitting it: it
    # tells GDAL the source has no nodata, overriding the value the file
    # declares. The HM rasters declare +3.4e38, so that silently turned ocean
    # into valid data and the land mask became the whole grid.
    override = {} if src_nodata is None else {'src_nodata': src_nodata}
    return stack.enter_context(WarpedVRT(
        src, crs=config.EQUAL_AREA_CRS, transform=transform,
        width=width, height=height,
        resampling=getattr(Resampling, config.EQUAL_AREA_RESAMPLING),
        nodata=fill, **override,
    ))


def read_masked(src, window=None) -> np.ndarray:
    """Read as float32 with every nodata convention collapsed to NaN.

    Some rasters declare nodata (+/-3.4e38) while others carry bare NaN in the
    pixel data. masked=True catches only the former, so the mask is then filled
    with NaN to put both on the same footing. Without this, 3.4e38 passes every
    '> threshold' test as a spurious increase - and ocean minus ocean comes out
    as exactly 0.0, which reads as a genuine "no change" observation.
    """
    band = src.read(1, window=window, masked=True)
    return np.ma.filled(band.astype(np.float32), np.nan)
