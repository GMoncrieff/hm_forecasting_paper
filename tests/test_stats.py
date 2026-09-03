"""Fixture-based checks for the streaming counters in stats.py.

Two independent guards:
  * hand-computed expectations for the observed-change and central-2040 blocks
  * a whole-array reference implementation compared against the streamed result
    for all twelve forecasts, run at several strip heights
"""
import contextlib

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import config
import stats

ND = np.float32(3.4e38)   # the declared nodata the observed rasters use

# 4x4 fixtures. 'ND' marks declared nodata; np.nan marks in-band NaN.
OBS_2020 = np.array([
    [0.00, 0.05, 0.09,   ND],
    [0.10, 0.20, 0.39, 0.40],
    [0.50, 0.95, 0.02, 0.15],
    [  ND,   ND, 0.30, 0.08],
], dtype=np.float32)

OBS_2000 = np.array([
    [0.00, 0.04, 0.09,   ND],
    [0.11, 0.20, 0.30, 0.50],
    [0.50, 0.90, 0.02, 0.15],
    [  ND,   ND, 0.30, 0.00],
], dtype=np.float32)

# Forecasts carry NaN (not ND) wherever they have no value, as the real ones do.
CENTRAL_2040 = np.array([
    [0.00, 0.20,    0.09, np.nan],
    [0.10, 0.50,    0.39,   0.42],
    [0.50, 0.95,    0.11,   0.15],
    [np.nan, np.nan, 0.30,  0.08],
], dtype=np.float32)


def _write(path, array, transform=None):
    with rasterio.open(
        path, 'w', driver='GTiff', height=array.shape[0], width=array.shape[1],
        count=1, dtype='float32', crs='EPSG:4326',
        transform=transform or from_origin(-180, 84, 0.009, 0.009),
        nodata=float(ND),
    ) as dst:
        dst.write(array, 1)


@pytest.fixture
def fixture_rasters(tmp_path, monkeypatch):
    """Point config.PATHS at small rasters and return the arrays behind them.

    Counting runs on the native grid here: these fixtures are 4x4 cells, so a
    warp would resample away the hand-computed expectations. The equal-area
    path is covered separately by test_equal_area_reweights_by_latitude.
    """
    monkeypatch.setattr(config, 'EQUAL_AREA_CRS', None)
    rng = np.random.default_rng(0)
    absent = ~np.isfinite(CENTRAL_2040)          # shared no-data footprint

    arrays = {'hm_2000_aa': OBS_2000, 'hm_2020_aa': OBS_2020}
    for scenario in stats.SCENARIOS:
        for year in stats.FORECAST_YEARS:
            key = stats.forecast_key(scenario, year)
            if (scenario, year) == ('central', 2040):
                arrays[key] = CENTRAL_2040
                continue
            a = rng.uniform(0.0, 0.6, size=OBS_2020.shape).astype(np.float32)
            a[absent] = np.nan
            arrays[key] = a

    paths = dict(config.PATHS)
    for key, array in arrays.items():
        path = tmp_path / f'{key}.tif'
        _write(path, array)
        paths[key] = path
    monkeypatch.setattr(config, 'PATHS', paths)
    return arrays


def _finite(a):
    """Valid mask under both nodata conventions, matching stats.read_strip."""
    return np.isfinite(a) & (a != ND)


def reference(arrays):
    """Whole-array reference for what collect() accumulates strip by strip."""
    obs2000, obs2020 = arrays['hm_2000_aa'], arrays['hm_2020_aa']
    f2000, f2020 = _finite(obs2000), _finite(obs2020)
    low_base = f2020 & (obs2020 < stats.LOW_CUT)
    mid_base = f2020 & (obs2020 >= stats.LOW_CUT) & (obs2020 < stats.HIGH_CUT)

    out = {'valid_2020': int(f2020.sum()), 'low': {}, 'mid': {}, 'change': {}}

    both = f2000 & f2020
    d = np.where(both, obs2020 - obs2000, 0.0)
    out['change']['observed'] = {
        t: dict(increase=int((both & (d > t)).sum()),
                decrease=int((both & (d < -t)).sum()),
                n=int(both.sum()))
        for t in stats.CHANGE_THRESHOLDS}

    for scenario in stats.SCENARIOS:
        for year in stats.FORECAST_YEARS:
            pred = arrays[stats.forecast_key(scenario, year)]
            fin = _finite(pred)
            lb, mb = low_base & fin, mid_base & fin
            out['low'][(scenario, year)] = dict(
                base=int(lb.sum()),
                crossed=int((lb & (pred >= stats.LOW_CUT)).sum()))
            out['mid'][(scenario, year)] = dict(
                base=int(mb.sum()),
                crossed=int((mb & (pred >= stats.HIGH_CUT)).sum()))
            if (scenario, year) == ('central', 2040):
                ok = f2020 & fin
                d = np.where(ok, pred - obs2020, 0.0)
                out['change']['projected'] = {
                    t: dict(increase=int((ok & (d > t)).sum()),
                            decrease=int((ok & (d < -t)).sum()),
                            n=int(ok.sum()))
                    for t in stats.CHANGE_THRESHOLDS}
    return out


def test_hand_computed_change(fixture_rasters):
    res = stats.collect()
    assert res['valid_2020'] == 13

    t = stats.CHANGE_THRESHOLD          # 0.005, the headline threshold
    obs = res['change']['observed'][t]
    assert (obs['n'], obs['increase'], obs['decrease'], obs['stable']) == (13, 4, 2, 7)

    proj = res['change']['projected'][t]
    assert (proj['n'], proj['increase'], proj['decrease'], proj['stable']) == (13, 4, 0, 9)


def test_every_threshold_is_reported(fixture_rasters):
    res = stats.collect()
    for key in ('observed', 'projected'):
        assert set(res['change'][key]) == set(stats.CHANGE_THRESHOLDS), key


def test_raising_the_threshold_moves_pixels_into_stable(fixture_rasters):
    """A wider stable band can only shrink increase/decrease, never grow them.

    Exact counts at 0.01 are not asserted: the fixture's +/-0.01 deltas land on
    the cut in float32, so which side they fall is not a property worth pinning.
    """
    res = stats.collect()
    for key in ('observed', 'projected'):
        by_threshold = res['change'][key]
        rows = [by_threshold[t] for t in sorted(by_threshold)]
        for lo, hi in zip(rows, rows[1:]):
            assert hi['increase'] <= lo['increase'], key
            assert hi['decrease'] <= lo['decrease'], key
            assert hi['stable'] >= lo['stable'], key
        for row in rows:
            assert row['increase'] + row['decrease'] + row['stable'] == row['n']


def test_hand_computed_exceedance(fixture_rasters):
    res = stats.collect()
    low = res['low'][('central', 2040)]
    mid = res['mid'][('central', 2040)]
    assert (low['base'], low['crossed']) == (5, 2)    # 0.20 and 0.11 cross 0.10
    assert (mid['base'], mid['crossed']) == (5, 1)    # only 0.50 crosses 0.40

    # Unrestricted bases: 5 pixels < 0.10, 5 pixels in [0.10, 0.40)
    assert (res['low_base_all'], res['mid_base_all']) == (5, 5)
    assert all(v['base'] <= res['low_base_all'] for v in res['low'].values())
    assert all(v['base'] <= res['mid_base_all'] for v in res['mid'].values())


@pytest.mark.parametrize('rows_per_strip', [1, 2, 3, 4, 100])
def test_matches_reference_at_every_strip_height(
        fixture_rasters, monkeypatch, rows_per_strip):
    monkeypatch.setattr(stats, 'ROWS_PER_STRIP', rows_per_strip)
    res = stats.collect()
    ref = reference(fixture_rasters)

    assert res['valid_2020'] == ref['valid_2020']
    for key, by_threshold in ref['change'].items():
        for t, expected in by_threshold.items():
            got = res['change'][key][t]
            assert (got['increase'], got['decrease'], got['n']) == (
                expected['increase'], expected['decrease'],
                expected['n']), (key, t)
            assert got['stable'] == (expected['n'] - expected['increase']
                                     - expected['decrease'])
    for block in ('low', 'mid'):
        assert res[block] == ref[block], block


def test_declared_nodata_never_counts_as_increase(tmp_path, monkeypatch):
    """3.4e38 must be masked, not read as a giant HM rise."""
    arr = np.array([[0.10, ND]], dtype=np.float32)
    path = tmp_path / 'nd.tif'
    _write(path, arr)
    with rasterio.open(path) as src:
        strip = stats.read_strip(src, rasterio.windows.Window(0, 0, 2, 1))
    assert strip[0, 0] == pytest.approx(0.10)
    assert np.isnan(strip[0, 1])
    assert not np.any(strip > 1.0)


def _latitude_band_rasters(tmp_path, monkeypatch, res_m):
    """1-degree global raster, lat -60..60, with HM rising only above lat 30."""
    rows, cols = 120, 360                      # lat 60..-60, lon -180..180
    transform = from_origin(-180, 60, 1.0, 1.0)

    obs2000 = np.zeros((rows, cols), dtype=np.float32)
    obs2020 = np.zeros((rows, cols), dtype=np.float32)
    obs2020[:30, :] = 0.02                     # top 30 rows = lat 30..60

    arrays = {'hm_2000_aa': obs2000, 'hm_2020_aa': obs2020}
    for scenario in stats.SCENARIOS:
        for year in stats.FORECAST_YEARS:
            arrays[stats.forecast_key(scenario, year)] = obs2020

    paths = dict(config.PATHS)
    for key, array in arrays.items():
        path = tmp_path / f'{key}.tif'
        _write(path, array, transform=transform)
        paths[key] = path
    monkeypatch.setattr(config, 'PATHS', paths)
    monkeypatch.setattr(config, 'EQUAL_AREA_RES', res_m)
    monkeypatch.setattr(stats, 'ROWS_PER_STRIP', 256)


def test_equal_area_reweights_by_latitude(tmp_path, monkeypatch):
    """The lat 30-60 band is 25% of rows but only ~21% of the area.

    sin(60)-sin(30) over sin(60)-sin(-60) = 0.2113, so warping to an equal-area
    grid must pull the reported share down from 25% to about 21%.
    """
    _latitude_band_rasters(tmp_path, monkeypatch, res_m=25_000)

    monkeypatch.setattr(config, 'EQUAL_AREA_CRS', None)
    native = stats.collect()['change']['observed'][stats.CHANGE_THRESHOLD]
    native_pct = 100.0 * native['increase'] / native['n']

    monkeypatch.setattr(config, 'EQUAL_AREA_CRS', 'EPSG:6933')
    equal = stats.collect()['change']['observed'][stats.CHANGE_THRESHOLD]
    equal_pct = 100.0 * equal['increase'] / equal['n']

    expected = 100.0 * (np.sin(np.deg2rad(60)) - np.sin(np.deg2rad(30))) / (
        2 * np.sin(np.deg2rad(60)))

    assert native_pct == pytest.approx(25.0, abs=0.01)
    assert expected == pytest.approx(21.13, abs=0.01)
    assert equal_pct == pytest.approx(expected, abs=0.5)
    assert equal_pct < native_pct - 3.0        # the correction is substantial


@pytest.mark.parametrize('crs', ['EPSG:6933', 'ESRI:54009'])
def test_equal_area_result_is_projection_independent(tmp_path, monkeypatch, crs):
    """Equal-area CRSs whose domain the grid respects must agree."""
    _latitude_band_rasters(tmp_path, monkeypatch, res_m=25_000)
    monkeypatch.setattr(config, 'EQUAL_AREA_CRS', crs)
    res = stats.collect()
    row = res['change']['observed'][stats.CHANGE_THRESHOLD]
    assert 100.0 * row['increase'] / row['n'] == pytest.approx(21.13, abs=0.6)
    assert abs(stats.area_ratio(res)) < stats.AREA_TOLERANCE_PCT


def test_area_guard_accepts_a_sound_projection(tmp_path, monkeypatch):
    """EPSG:6933 must reproduce the true band area, so the guard stays quiet."""
    _latitude_band_rasters(tmp_path, monkeypatch, res_m=25_000)
    monkeypatch.setattr(config, 'EQUAL_AREA_CRS', 'EPSG:6933')
    res = stats.collect()

    r = stats.AUTHALIC_RADIUS_M
    true_km2 = 2 * np.pi * r ** 2 * 2 * np.sin(np.deg2rad(60)) / 1e6
    assert res['native_km2'] == pytest.approx(true_km2, rel=0.005)
    assert res['warped_km2'] == pytest.approx(true_km2, rel=0.01)
    assert abs(stats.area_ratio(res)) < stats.AREA_TOLERANCE_PCT


def test_area_guard_catches_overfilling_projection(tmp_path, monkeypatch):
    """Equal Earth's rectangular grid keeps out-of-domain corners.

    Its inverse wraps those corners onto real longitudes, so they fill with
    data instead of NaN and inflate the land area ~8%. The guard must notice;
    silently trusting it would bias every percentage in the report.
    """
    _latitude_band_rasters(tmp_path, monkeypatch, res_m=25_000)
    monkeypatch.setattr(config, 'EQUAL_AREA_CRS', 'EPSG:8857')
    res = stats.collect()
    assert stats.area_ratio(res) > stats.AREA_TOLERANCE_PCT
    assert res['warped_km2'] > res['native_km2'] * 1.05


def test_warped_layers_share_one_grid(tmp_path, monkeypatch):
    """All fourteen warps must land on an identical grid, or differencing lies."""
    _latitude_band_rasters(tmp_path, monkeypatch, res_m=50_000)
    monkeypatch.setattr(config, 'EQUAL_AREA_CRS', 'EPSG:6933')
    with contextlib.ExitStack() as stack:
        with rasterio.open(config.PATHS['hm_2020_aa']) as ref:
            grid = stats.target_grid(ref)
        layers = [stats.open_layer(stack, config.PATHS[k], grid)
                  for k in ('hm_2000_aa', 'hm_2020_aa',
                            stats.forecast_key('upper', 2040),
                            stats.forecast_key('lower', 2025))]
        shapes = {(v.height, v.width) for v in layers}
        transforms = {tuple(v.transform)[:6] for v in layers}
        crses = {v.crs.to_string() for v in layers}
    assert len(shapes) == 1 and len(transforms) == 1
    assert crses == {'EPSG:6933'}


def test_warp_fills_gaps_with_nan_not_zero(tmp_path, monkeypatch):
    """An unset destination nodata would fill with 0.0 and read as low-HM land."""
    _latitude_band_rasters(tmp_path, monkeypatch, res_m=25_000)
    monkeypatch.setattr(config, 'EQUAL_AREA_CRS', 'EPSG:6933')
    with contextlib.ExitStack() as stack:
        with rasterio.open(config.PATHS['hm_2020_aa']) as ref:
            grid = stats.target_grid(ref)
        vrt = stats.open_layer(stack, config.PATHS['hm_2020_aa'], grid)
        # The source spans lat -60..60; the snapped grid overshoots slightly, so
        # the first destination row must be outside the source footprint.
        top = stats.read_strip(vrt, rasterio.windows.Window(0, 0, vrt.width, 1))
    assert np.isnan(top).any(), 'warp gap should be NaN'
    assert not np.any(top == 0.0), 'warp gap must not be filled with 0.0'


def test_in_band_nan_survives_read(tmp_path):
    """Forecast rasters store NaN in-band; it must stay NaN, not become 0."""
    arr = np.array([[0.25, np.nan]], dtype=np.float32)
    path = tmp_path / 'nan.tif'
    _write(path, arr)
    with rasterio.open(path) as src:
        strip = stats.read_strip(src, rasterio.windows.Window(0, 0, 2, 1))
    assert strip[0, 0] == pytest.approx(0.25)
    assert np.isnan(strip[0, 1])
