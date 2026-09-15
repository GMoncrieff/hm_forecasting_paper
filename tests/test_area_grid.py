"""Checks for the shared equal-area helpers in utils.py.

tests/test_stats.py already exercises the warp through stats.py's aliases; this
file covers what the move to utils added - the integer-raster guard, dtype
preservation, and the area-fair quantile fitting the Fig 9 colour scale uses.
"""
import contextlib

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin

import config
import stats
import utils

ND = np.float32(3.4e38)


def _write(path, array, *, transform=None, nodata=float(ND), dtype='float32'):
    with rasterio.open(
        path, 'w', driver='GTiff', height=array.shape[0], width=array.shape[1],
        count=1, dtype=dtype, crs='EPSG:4326',
        transform=transform or from_origin(-180, 60, 1.0, 1.0), nodata=nodata,
    ) as dst:
        dst.write(array.astype(dtype), 1)


@pytest.fixture
def latitude_band(tmp_path, monkeypatch):
    """A 1-degree global raster, lat -60..60, values rising above lat 30."""
    monkeypatch.setattr(config, 'EQUAL_AREA_CRS', 'EPSG:6933')
    monkeypatch.setattr(config, 'EQUAL_AREA_RES', 25_000)
    arr = np.zeros((120, 360), dtype=np.float32)
    arr[:30, :] = 1.0                      # top 30 rows = lat 30..60
    path = tmp_path / 'band.tif'
    _write(path, arr)
    return path


def test_stats_reuses_the_utils_helpers():
    """One implementation, not two that can drift apart."""
    assert stats.target_grid is utils.equal_area_grid
    assert stats.open_layer is utils.open_equal_area
    assert stats.read_strip is utils.read_masked
    assert stats.AUTHALIC_RADIUS_M == utils.AUTHALIC_RADIUS_M


def test_grid_is_none_when_disabled(latitude_band, monkeypatch):
    monkeypatch.setattr(config, 'EQUAL_AREA_CRS', None)
    with rasterio.open(latitude_band) as src:
        assert utils.equal_area_grid(src) is None


def test_grid_is_none_passes_the_raster_through(latitude_band, monkeypatch):
    monkeypatch.setattr(config, 'EQUAL_AREA_CRS', None)
    with contextlib.ExitStack() as stack:
        layer = utils.open_equal_area(stack, latitude_band, None)
        assert layer.crs.to_string() == 'EPSG:4326'
        assert layer.shape == (120, 360)


def test_warp_reweights_a_latitude_band(latitude_band):
    """25% of rows is only 21.13% of the area: sin60-sin30 over 2*sin60."""
    with rasterio.open(latitude_band) as ref:
        grid = utils.equal_area_grid(ref)
    with contextlib.ExitStack() as stack:
        vrt = utils.open_equal_area(stack, latitude_band, grid)
        a = utils.read_masked(vrt)

    fin = np.isfinite(a)
    share = 100.0 * np.count_nonzero(fin & (a > 0.5)) / np.count_nonzero(fin)
    expected = 100.0 * (np.sin(np.deg2rad(60)) - np.sin(np.deg2rad(30))) / (
        2 * np.sin(np.deg2rad(60)))
    assert expected == pytest.approx(21.13, abs=0.01)
    assert share == pytest.approx(expected, abs=0.5)
    assert share < 25.0 - 3.0     # clearly moved off the pixel-count answer


def test_warp_preserves_the_declared_nodata(tmp_path, monkeypatch):
    """A raster that *contains* its declared nodata must keep it masked.

    Regression: open_equal_area passed src_nodata=None explicitly, which tells
    GDAL the source has no nodata and overrides what the file declares. The HM
    rasters declare +3.4e38, so ocean came back as valid data and the land mask
    became the entire grid. Every earlier fixture either skipped the warp or
    held no nodata cells, so nothing caught it.
    """
    monkeypatch.setattr(config, 'EQUAL_AREA_CRS', 'EPSG:6933')
    monkeypatch.setattr(config, 'EQUAL_AREA_RES', 25_000)
    arr = np.full((120, 360), float(ND), dtype=np.float32)
    arr[40:80, :] = 0.5                      # a valid band amid declared nodata
    path = tmp_path / 'holes.tif'
    _write(path, arr)

    with rasterio.open(path) as ref:
        grid = utils.equal_area_grid(ref)
    with contextlib.ExitStack() as stack:
        vrt = utils.open_equal_area(stack, path, grid)
        assert vrt.src_nodata == pytest.approx(float(ND)), \
            'the source nodata was overridden'
        out = utils.read_masked(vrt)

    finite = np.isfinite(out)
    assert not np.any(out > 1.0), '3.4e38 leaked through as data'
    assert 0.05 < finite.mean() < 0.95, \
        f'expected a partial mask, got {finite.mean():.1%} valid'
    assert np.allclose(out[finite], 0.5)


def test_explicit_src_nodata_still_overrides(tmp_path, monkeypatch):
    """Passing src_nodata must still win for files that misstate or omit it."""
    monkeypatch.setattr(config, 'EQUAL_AREA_CRS', 'EPSG:6933')
    monkeypatch.setattr(config, 'EQUAL_AREA_RES', 25_000)
    arr = np.full((120, 360), 7.0, dtype=np.float32)
    arr[40:80, :] = 0.5
    path = tmp_path / 'wrong_nodata.tif'
    _write(path, arr, nodata=None)           # file declares nothing

    with rasterio.open(path) as ref:
        grid = utils.equal_area_grid(ref)
    with contextlib.ExitStack() as stack:
        vrt = utils.open_equal_area(stack, path, grid, src_nodata=7.0)
        out = utils.read_masked(vrt)
    assert not np.any(out == 7.0), 'explicit src_nodata was ignored'
    assert np.allclose(out[np.isfinite(out)], 0.5)


def test_integer_raster_without_fill_is_refused(tmp_path, monkeypatch):
    """A NaN fill on uint8 promotes the VRT to float64; make callers choose."""
    monkeypatch.setattr(config, 'EQUAL_AREA_CRS', 'EPSG:6933')
    monkeypatch.setattr(config, 'EQUAL_AREA_RES', 25_000)
    path = tmp_path / 'classes.tif'
    _write(path, np.arange(120 * 360).reshape(120, 360) % 5,
           nodata=None, dtype='uint8')
    with rasterio.open(path) as ref:
        grid = utils.equal_area_grid(ref)
    with contextlib.ExitStack() as stack:
        with pytest.raises(ValueError, match='fill='):
            utils.open_equal_area(stack, path, grid)


def test_integer_raster_with_sentinel_keeps_its_dtype(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'EQUAL_AREA_CRS', 'EPSG:6933')
    monkeypatch.setattr(config, 'EQUAL_AREA_RES', 25_000)
    path = tmp_path / 'classes.tif'
    _write(path, np.arange(120 * 360).reshape(120, 360) % 5,
           nodata=None, dtype='uint8')
    with rasterio.open(path) as ref:
        grid = utils.equal_area_grid(ref)
    with contextlib.ExitStack() as stack:
        vrt = utils.open_equal_area(stack, path, grid, src_nodata=255, fill=255)
        assert vrt.dtypes[0] == 'uint8'
        assert set(np.unique(vrt.read(1))) <= {0, 1, 2, 3, 4, 255}


def test_read_masked_collapses_both_nodata_conventions(tmp_path):
    """+3.4e38 and in-band NaN must both come back as NaN, never as a value."""
    arr = np.array([[0.25, float(ND), np.nan]], dtype=np.float32)
    path = tmp_path / 'mixed.tif'
    _write(path, arr)
    with rasterio.open(path) as src:
        out = utils.read_masked(src)
    assert out[0, 0] == pytest.approx(0.25)
    assert np.isnan(out[0, 1]) and np.isnan(out[0, 2])
    assert not np.any(out > 1.0)


# --- Fig 9 colour scale -----------------------------------------------------

def test_quantile_fit_uses_the_whole_basis_not_a_subsample():
    """sklearn's QuantileTransformer defaults to subsample=10_000."""
    rng = np.random.default_rng(0)
    basis = rng.normal(size=50_000)
    da = xr.DataArray(basis.astype(np.float32).reshape(200, 250))
    a = utils.transform_xarray_layer(da, fit_values=basis)
    b = utils.transform_xarray_layer(da, fit_values=basis)
    # Identical across runs: no hidden random subsample of the fit basis.
    assert np.allclose(a.values, b.values, equal_nan=True)


def test_quantile_fit_basis_changes_the_mapping():
    """Fitting on a different population must move the colour breaks."""
    da = xr.DataArray(np.linspace(0, 1, 10_000, dtype=np.float32).reshape(100, 100))
    low = utils.transform_xarray_layer(da, fit_values=np.linspace(0, 0.2, 5_000))
    full = utils.transform_xarray_layer(da, fit_values=np.linspace(0, 1.0, 5_000))
    assert not np.allclose(low.values, full.values, equal_nan=True)


def test_transform_preserves_rank_order():
    da = xr.DataArray(np.array([[0.1, 0.5, 0.9], [0.2, np.nan, 0.7]],
                               dtype=np.float32))
    out = utils.transform_xarray_layer(da, fit_values=np.linspace(0, 1, 1000))
    v, o = da.values.ravel(), out.values.ravel()
    ok = np.isfinite(v) & np.isfinite(o)
    assert np.array_equal(np.argsort(v[ok]), np.argsort(o[ok]))


def test_ternary_thresholds_follow_the_fit_values(capsys):
    """fit_values must drive the opacity cut-points, not the rendered array."""
    shape = (40, 60)
    base = np.linspace(0.0, 1.0, shape[0] * shape[1],
                       dtype=np.float32).reshape(shape)
    ds = xr.Dataset({k: (('y', 'x'), base.copy()) for k in ('a', 'b', 'c')})

    small = {k: np.linspace(0.0, 0.1, 5_000, dtype=np.float32) for k in 'abc'}
    large = {k: np.linspace(0.0, 1.0, 5_000, dtype=np.float32) for k in 'abc'}

    utils.create_ternary_alpha_array(ds, 'a', 'b', 'c', fit_values=small)
    lo = capsys.readouterr().out
    utils.create_ternary_alpha_array(ds, 'a', 'b', 'c', fit_values=large)
    hi = capsys.readouterr().out

    assert 'equal-area sample' in lo and 'equal-area sample' in hi
    assert lo != hi, 'cut-points ignored the fit basis'


def test_ternary_without_fit_values_warns_it_is_not_area_fair(capsys):
    shape = (40, 60)
    base = np.linspace(0.0, 1.0, shape[0] * shape[1],
                       dtype=np.float32).reshape(shape)
    ds = xr.Dataset({k: (('y', 'x'), base.copy()) for k in ('a', 'b', 'c')})
    utils.create_ternary_alpha_array(ds, 'a', 'b', 'c')
    assert 'NOT area-fair' in capsys.readouterr().out


# --- antimeridian trim ------------------------------------------------------
#
# equal_area_grid snaps outward to whole cells, so the grid is slightly wider
# than the projection's valid domain and its edge columns straddle +/-180.
# Cartopy draws such a quad across the whole map, which put a stripe through
# Figure 10 at every latitude with land near the date line.


def test_trim_drops_exactly_one_column_at_each_edge():
    import rasterio
    with rasterio.open(config.PATHS['hm_static_iucn_strict']) as ref:
        transform, width, _ = utils.equal_area_grid(ref)
    for stride in (1, 4, 8):
        x = transform.c + transform.a * stride * (np.arange(width // stride) + 0.5)
        data = np.zeros((3, x.size))
        x2, d2 = utils.trim_wrapped_columns(x, data)
        assert x.size - x2.size == 2, stride
        assert d2.shape == (3, x2.size)
        assert np.array_equal(x2, x[1:-1])


def test_trim_keeps_a_grid_that_is_already_inside_the_domain():
    """A grid comfortably inside the valid x range must not lose anything."""
    x = np.linspace(-1e6, 1e6, 50)
    data = np.zeros((2, 50))
    x2, d2 = utils.trim_wrapped_columns(x, data)
    assert np.array_equal(x2, x)
    assert d2.shape == data.shape


def test_trim_is_a_no_op_without_an_equal_area_crs(monkeypatch):
    monkeypatch.setattr(config, 'EQUAL_AREA_CRS', None)
    x = np.linspace(-2e7, 2e7, 10)
    data = np.zeros((1, 10))
    x2, d2 = utils.trim_wrapped_columns(x, data)
    assert np.array_equal(x2, x) and d2.shape == data.shape


def test_trim_handles_several_arrays_and_leading_axes():
    x = np.array([-17366000.0, -17362000.0, 0.0, 17362000.0, 17366000.0])
    a = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    b = np.ones((5,), dtype=bool)
    x2, a2, b2 = utils.trim_wrapped_columns(x, a, b)
    assert x2.size == 3
    assert a2.shape == (2, 3, 3)
    assert b2.shape == (3,)
