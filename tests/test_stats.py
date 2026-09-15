"""Fixture-based checks for the streaming accumulators in stats.py.

Three independent guards:
  * hand-computed expectations for the observed-change block and for both
    expected-area blocks, including the two dependence brackets
  * a whole-array reference implementation compared against the streamed result
    at several strip heights
  * the equal-area behaviour of the warp, which is what makes a sum of
    probabilities times cell area an area at all
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

OBS_1990 = np.array([
    [0.00, 0.04, 0.08,   ND],
    [0.05, 0.15, 0.25, 0.30],
    [0.45, 0.85, 0.01, 0.12],
    [  ND,   ND, 0.28, 0.00],
], dtype=np.float32)

OBS_2010 = np.array([
    [0.00, 0.05, 0.09,   ND],
    [0.10, 0.20, 0.35, 0.45],
    [0.50, 0.92, 0.02, 0.14],
    [  ND,   ND, 0.29, 0.05],
], dtype=np.float32)

# Probability layers carry NaN (not ND) where they have no value, as the real
# COGs do. The NaN at (0,2) sits inside the natural base on purpose: it is what
# makes the per-year denominator differ from the unrestricted base.
P10_2040 = np.array([
    [0.10, 0.20, np.nan, np.nan],
    [0.30, 0.40,   0.50,   0.60],
    [0.70, 0.80,   0.99,   0.99],
    [np.nan, np.nan, 0.55,  0.02],
], dtype=np.float32)

# Monotone quantile function => p40 <= p10 everywhere.
P40_2040 = np.array([
    [0.05, 0.10, np.nan, np.nan],
    [0.15, 0.20,   0.25,   0.30],
    [0.35, 0.40,   0.45,   0.98],
    [np.nan, np.nan, 0.28,  0.01],
], dtype=np.float32)

MEAN_2040 = np.array([
    [0.00, 0.20,   0.09, np.nan],
    [0.10, 0.50,   0.39,   0.42],
    [0.50, 0.95,   0.11,   0.15],
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
    absent = ~np.isfinite(P10_2040)          # shared no-data footprint

    # hm_1990_aa must be in the fixture: without it config.PATHS still points at
    # the real 837 MB raster and collect() silently mixes real data into a 4x4
    # test, which passes while measuring nothing.
    arrays = {'hm_1990_aa': OBS_1990, 'hm_2000_aa': OBS_2000,
              'hm_2010_aa': OBS_2010, 'hm_2020_aa': OBS_2020}
    last = stats.FORECAST_YEARS[-1]
    for year in stats.FORECAST_YEARS:
        if year == last:
            arrays[f'hm_p10_{year}'] = P10_2040
            arrays[f'hm_p40_{year}'] = P40_2040
            arrays[stats.mean_key(year)] = MEAN_2040
            continue
        p10 = rng.uniform(0.0, 1.0, size=OBS_2020.shape).astype(np.float32)
        p40 = (p10 * rng.uniform(0.0, 1.0, size=OBS_2020.shape)).astype(np.float32)
        mean = rng.uniform(0.0, 0.6, size=OBS_2020.shape).astype(np.float32)
        for a in (p10, p40, mean):
            a[absent] = np.nan
        arrays[f'hm_p10_{year}'] = p10
        arrays[f'hm_p40_{year}'] = p40
        arrays[stats.mean_key(year)] = mean

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
    # Cuts closed at the bottom: natural is HM <= LOW_CUT. See config.LOW_CUT.
    natural_base = f2020 & (obs2020 <= stats.LOW_CUT)
    moderate_base = (f2020 & (obs2020 > stats.LOW_CUT)
                     & (obs2020 < stats.HIGH_CUT))

    out = {'valid_2020': int(f2020.sum()), 'natural': {}, 'moderate': {},
           'budget': {}, 'added': {}, 'change': {}}

    both = f2000 & f2020
    d = np.where(both, obs2020 - obs2000, 0.0)
    out['change'] = {
        t: dict(increase=int((both & (d > t)).sum()),
                decrease=int((both & (d < -t)).sum()),
                n=int(both.sum()))
        for t in stats.CHANGE_THRESHOLDS}
    out['observed_added'] = dict(n=int(both.sum()),
                                 total=float(d[both].sum(dtype=np.float64)))

    def block(base, p):
        v = p[base & _finite(p)].astype(np.float64)
        return dict(base=int(v.size), sum_p=float(v.sum()),
                    sum_pq=float((v * (1.0 - v)).sum()),
                    n_hi=int((v > 0.975).sum()), n_lo=int((v > 0.025).sum()))

    for year in stats.FORECAST_YEARS:
        p10 = arrays[f'hm_p10_{year}']
        p40 = arrays[f'hm_p40_{year}']
        mean = arrays[stats.mean_key(year)]
        out['natural'][year] = block(natural_base, p10)
        out['moderate'][year] = block(moderate_base, p40)

        ok = f2020 & _finite(p10) & _finite(p40)
        a, b = p10[ok].astype(np.float64), p40[ok].astype(np.float64)
        out['budget'][year] = dict(
            n=int(ok.sum()), natural=float((1.0 - a).sum()),
            moderate=float((a - b).sum()), high=float(b.sum()),
            inverted=int((b > a).sum()))

        okm = f2020 & _finite(mean)
        out['added'][year] = dict(
            n=int(okm.sum()),
            total=float((mean[okm].astype(np.float64) - obs2020[okm]).sum()))
    return out


def test_hand_computed_change(fixture_rasters):
    """The observed-change block is unchanged and must still hold.

    13 cells are finite in both years (three carry ND in both). At the 0.005
    threshold: (0,1) +0.01, (1,2) +0.09, (2,1) +0.05 and (3,3) +0.08 rise;
    (1,0) -0.01 and (1,3) -0.10 fall; the other seven are unchanged.
    """
    row = stats.collect()['change'][stats.CHANGE_THRESHOLDS[0]]
    assert row['n'] == 13
    assert row['increase'] == 4
    assert row['decrease'] == 2
    assert row['stable'] == 7


def test_hand_computed_natural_expected_area(fixture_rasters):
    """Base, expected sum and both brackets, computed by hand.

    Natural base is HM 2020 <= 0.10: cells (0,0)=0.00, (0,1)=0.05, (0,2)=0.09,
    (1,0)=0.10, (2,2)=0.02, (3,3)=0.08 -> 6 cells. (1,0) sits exactly on the
    cut, which is the case the closed cut exists to settle. p10 is NaN at
    (0,2), so this year's denominator is 5, and the probabilities are 0.10,
    0.20, 0.30, 0.99, 0.02.
    """
    res = stats.collect()
    year = stats.FORECAST_YEARS[-1]
    block = res['natural'][year]

    assert res['natural_base_all'] == 6
    assert block['base'] == 5                      # the NaN drops one
    assert block['sum_p'] == pytest.approx(0.10 + 0.20 + 0.30 + 0.99 + 0.02)
    assert block['sum_pq'] == pytest.approx(
        0.10 * 0.90 + 0.20 * 0.80 + 0.30 * 0.70
        + 0.99 * 0.01 + 0.02 * 0.98)
    assert block['n_hi'] == 1                      # only 0.99 > 0.975
    assert block['n_lo'] == 4                      # 0.02 is below 0.025


def test_hand_computed_moderate_expected_area(fixture_rasters):
    """Moderate base is 0.10 < HM 2020 < 0.40: four cells, all with p40.

    (1,0)=0.10 is excluded - it is natural under the closed low cut, which is
    what keeps the two bases disjoint.
    """
    res = stats.collect()
    block = res['moderate'][stats.FORECAST_YEARS[-1]]

    assert res['moderate_base_all'] == 4
    assert block['base'] == 4
    assert block['sum_p'] == pytest.approx(0.20 + 0.25 + 0.98 + 0.28)
    assert block['n_hi'] == 1                      # only 0.98 > 0.975
    assert block['n_lo'] == 4


def test_hand_computed_observed_loss(fixture_rasters):
    """Realised crossings over the observed record, on each base year's classes.

    From the 1990 base, six cells are below 0.10 and one of them - (1,0), which
    goes 0.05 -> 0.10 - actually crosses. Five cells sit in 0.10-0.40 and one,
    (1,3) at 0.30 -> 0.40, crosses the upper cut. From the 2000 base nothing
    crosses either cut.
    """
    obs = stats.collect()['observed_loss']
    assert obs[(1990, 'natural')] == {'base': 6, 'lost': 1}
    assert obs[(1990, 'moderate')] == {'base': 5, 'lost': 1}
    assert obs[(2000, 'natural')] == {'base': 5, 'lost': 0}
    assert obs[(2000, 'moderate')] == {'base': 5, 'lost': 0}


def test_observed_loss_never_exceeds_its_base(fixture_rasters):
    res = stats.collect()
    for name in ('observed_loss', 'observed_decade'):
        for key, row in res[name].items():
            assert 0 <= row['lost'] <= row['base'], (name, key)


def test_decades_cover_the_observed_record(fixture_rasters):
    """Three consecutive decades, each on its own start year's classes."""
    dec = stats.collect()['observed_decade']
    spans = sorted({k[0] for k in dec})
    assert spans == [(1990, 2000), (2000, 2010), (2010, 2020)]
    assert {k[1] for k in dec} == {'natural', 'moderate'}


def test_perfect_dependence_bracket_is_ordered(fixture_rasters):
    """p > 0.975 is a subset of p > 0.025, so the bracket cannot invert."""
    res = stats.collect()
    for name in ('natural', 'moderate'):
        for year, block in res[name].items():
            assert block['n_hi'] <= block['n_lo'], (name, year)
            assert block['n_lo'] <= block['base'], (name, year)


def test_condition_class_budget_closes(fixture_rasters):
    """natural + moderate + high must equal the land cell count, exactly."""
    res = stats.collect()
    for year, b in res['budget'].items():
        total = b['natural'] + b['moderate'] + b['high']
        assert total == pytest.approx(b['n'], rel=1e-12), year


def test_monotone_fixtures_report_no_inversions(fixture_rasters):
    """p40 <= p10 in the fixtures, so the spline-head warning must stay silent."""
    res = stats.collect()
    assert all(b['inverted'] == 0 for b in res['budget'].values())


def test_inverted_probabilities_are_counted(tmp_path, monkeypatch):
    """A p40 above p10 is reported, never clamped away."""
    monkeypatch.setattr(config, 'EQUAL_AREA_CRS', None)
    obs = np.array([[0.5, 0.5]], dtype=np.float32)
    p10 = np.array([[0.20, 0.40]], dtype=np.float32)
    p40 = np.array([[0.30, 0.10]], dtype=np.float32)   # first cell inverted

    paths = dict(config.PATHS)
    arrays = {'hm_2000_aa': obs, 'hm_2020_aa': obs}
    for year in stats.FORECAST_YEARS:
        arrays[f'hm_p10_{year}'] = p10
        arrays[f'hm_p40_{year}'] = p40
        arrays[stats.mean_key(year)] = obs
    for key, array in arrays.items():
        path = tmp_path / f'{key}.tif'
        _write(path, array)
        paths[key] = path
    monkeypatch.setattr(config, 'PATHS', paths)

    res = stats.collect()
    assert all(b['inverted'] == 1 for b in res['budget'].values())


def test_footprint_gap_is_reported(fixture_rasters):
    """p10 is NaN on one land cell; the report must say so rather than assume."""
    res = stats.collect()
    year = stats.FORECAST_YEARS[-1]
    # (0,2) and (3,0)/(3,1) are ND in HM 2020, so only (0,2) counts as a gap.
    assert res['gaps'][('p10', year)] == 1
    assert res['gaps'][('p40', year)] == 1


def test_expected_modification_added_matches_the_mean_layer(fixture_rasters):
    res = stats.collect()
    ref = reference(fixture_rasters)
    for year in stats.FORECAST_YEARS:
        assert res['added'][year]['total'] == pytest.approx(
            ref['added'][year]['total'], rel=1e-6)
        assert res['added'][year]['n'] == ref['added'][year]['n']
    assert res['observed_added']['total'] == pytest.approx(
        ref['observed_added']['total'], rel=1e-6)


def test_every_threshold_is_reported(fixture_rasters):
    res = stats.collect()
    assert set(res['change']) == set(stats.CHANGE_THRESHOLDS)
    assert set(res['expected_surface']) == set(stats.CHANGE_THRESHOLDS)


def test_raising_the_threshold_moves_pixels_into_stable(fixture_rasters):
    change = stats.collect()['change']
    ordered = sorted(stats.CHANGE_THRESHOLDS)
    stable = [change[t]['stable'] for t in ordered]
    assert stable == sorted(stable)


@pytest.mark.parametrize('rows_per_strip', [1, 2, 3, 4, 100])
def test_matches_reference_at_every_strip_height(
        fixture_rasters, monkeypatch, rows_per_strip):
    monkeypatch.setattr(stats, 'ROWS_PER_STRIP', rows_per_strip)
    res = stats.collect()
    ref = reference(fixture_rasters)

    assert res['valid_2020'] == ref['valid_2020']
    for t, expected in ref['change'].items():
        got = res['change'][t]
        assert (got['increase'], got['decrease'], got['n']) == (
            expected['increase'], expected['decrease'], expected['n']), t
        assert got['stable'] == (expected['n'] - expected['increase']
                                 - expected['decrease'])

    for name in ('natural', 'moderate'):
        for year, expected in ref[name].items():
            got = res[name][year]
            assert got['base'] == expected['base'], (name, year)
            assert got['n_hi'] == expected['n_hi'], (name, year)
            assert got['n_lo'] == expected['n_lo'], (name, year)
            assert got['sum_p'] == pytest.approx(expected['sum_p'], rel=1e-6)
            assert got['sum_pq'] == pytest.approx(expected['sum_pq'], rel=1e-6)
            # the histogram must account for every pixel in the base
            assert int(got['hist'].sum()) == expected['base']

    for year, expected in ref['budget'].items():
        got = res['budget'][year]
        assert got['n'] == expected['n']
        assert got['inverted'] == expected['inverted']
        for key in ('natural', 'moderate', 'high'):
            assert got[key] == pytest.approx(expected[key], rel=1e-6), (year, key)


def test_concentration_share_is_a_percentage(fixture_rasters):
    res = stats.collect()
    for year, block in res['natural'].items():
        share = stats.concentration_share(block)
        assert 0.0 <= share <= 100.0, year


def test_concentration_share_saturates_when_loss_is_concentrated():
    """One certain pixel among many near-zero ones carries almost all the loss."""
    block = stats._new_exceedance_block()
    p = np.concatenate([np.full(99, 0.001, dtype=np.float32),
                        np.array([1.0], dtype=np.float32)])
    stats._accumulate_exceedance(block, p)
    assert stats.concentration_share(block) > 90.0


def test_concentration_share_is_near_a_tenth_when_loss_is_uniform():
    """Identical probabilities put exactly a tenth of the loss in a tenth."""
    block = stats._new_exceedance_block()
    stats._accumulate_exceedance(block, np.full(1000, 0.3, dtype=np.float32))
    assert stats.concentration_share(block) == pytest.approx(10.0, abs=1.0)


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
    for year in stats.FORECAST_YEARS:
        arrays[f'hm_p10_{year}'] = obs2020
        arrays[f'hm_p40_{year}'] = np.zeros((rows, cols), dtype=np.float32)
        arrays[stats.mean_key(year)] = obs2020

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
    t = stats.CHANGE_THRESHOLDS[0]

    monkeypatch.setattr(config, 'EQUAL_AREA_CRS', None)
    native = stats.collect()['change'][t]
    native_pct = 100.0 * native['increase'] / native['n']

    monkeypatch.setattr(config, 'EQUAL_AREA_CRS', 'EPSG:6933')
    equal = stats.collect()['change'][t]
    equal_pct = 100.0 * equal['increase'] / equal['n']

    expected = 100.0 * (np.sin(np.deg2rad(60)) - np.sin(np.deg2rad(30))) / (
        2 * np.sin(np.deg2rad(60)))

    assert native_pct == pytest.approx(25.0, abs=0.01)
    assert expected == pytest.approx(21.13, abs=0.01)
    assert equal_pct == pytest.approx(expected, abs=0.5)
    assert equal_pct < native_pct - 3.0        # the correction is substantial


def test_equal_area_reweights_the_expected_area_too(tmp_path, monkeypatch):
    """The same correction must reach the probability sums, not just the counts.

    This is the reason equal-area matters more now rather than less: an
    expected area is a sum of probabilities times cell area, so it is only an
    area if the cells are.
    """
    _latitude_band_rasters(tmp_path, monkeypatch, res_m=25_000)
    year = stats.FORECAST_YEARS[-1]

    monkeypatch.setattr(config, 'EQUAL_AREA_CRS', None)
    native = stats.collect()['natural'][year]
    native_share = native['sum_p'] / native['base']

    monkeypatch.setattr(config, 'EQUAL_AREA_CRS', 'EPSG:6933')
    equal = stats.collect()['natural'][year]
    equal_share = equal['sum_p'] / equal['base']

    # p10 is 0.02 above lat 30 and 0 below, so the mean probability over the
    # base tracks the band's share of ground exactly as the counts did.
    assert native_share == pytest.approx(0.02 * 0.25, rel=0.01)
    assert equal_share == pytest.approx(0.02 * 0.2113, rel=0.05)


@pytest.mark.parametrize('crs', ['EPSG:6933', 'ESRI:54009'])
def test_equal_area_result_is_projection_independent(tmp_path, monkeypatch, crs):
    """Equal-area CRSs whose domain the grid respects must agree."""
    _latitude_band_rasters(tmp_path, monkeypatch, res_m=25_000)
    monkeypatch.setattr(config, 'EQUAL_AREA_CRS', crs)
    res = stats.collect()
    row = res['change'][stats.CHANGE_THRESHOLDS[0]]
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
    silently trusting it would bias every number in the report.
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
                  for k in ('hm_2000_aa', 'hm_2020_aa', 'hm_p10_2040',
                            'hm_p40_2025', stats.mean_key(2040))]
        shapes = {(v.height, v.width) for v in layers}
        transforms = {tuple(v.transform)[:6] for v in layers}
        crses = {v.crs.to_string() for v in layers}
    assert len(shapes) == 1 and len(transforms) == 1
    assert crses == {'EPSG:6933'}


def test_warp_fills_gaps_with_nan_not_zero(tmp_path, monkeypatch):
    """An unset destination nodata would fill with 0.0 and read as certainty."""
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
    """Probability rasters store NaN in-band; it must stay NaN, not become 0."""
    arr = np.array([[0.25, np.nan]], dtype=np.float32)
    path = tmp_path / 'nan.tif'
    _write(path, arr)
    with rasterio.open(path) as src:
        strip = stats.read_strip(src, rasterio.windows.Window(0, 0, 2, 1))
    assert strip[0, 0] == pytest.approx(0.25)
    assert np.isnan(strip[0, 1])


def test_report_runs_end_to_end(fixture_rasters, capsys):
    """Exercise report(), not just collect().

    The accumulator tests all call collect() directly, so a NameError inside
    report() - a local shadowing the module-level pct() helper, in the case that
    prompted this test - survived a green suite and only surfaced on a 6-minute
    production run.
    """
    stats.report(stats.collect())
    out = capsys.readouterr().out
    for heading in ('NATURAL LANDS LOST', 'MODERATELY MODIFIED LANDS LOST',
                    'DECADAL LOSS', 'CONDITION-CLASS BUDGET'):
        assert heading in out, heading
    for span in ('1990-2000', '2000-2010', '2010-2020', '2020-2030', '2030-2040'):
        assert span in out, span
