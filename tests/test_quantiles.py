"""Checks for quantiles.py - the quantile-function reader and its statistics.

The decode tests exist because the icechunk stores use `scale` and `sentinel`
rather than the CF names xarray decodes automatically. A value read without
decoding is ~32,000x too large and passes every threshold test silently, so
this is the one place a hard-coded constant would be actively dangerous.
"""
import numpy as np
import pytest
import xarray as xr

import quantiles as q


def _da(raw, scale, sentinel, offset=0.0):
    return xr.DataArray(
        np.asarray(raw, dtype=np.int16), dims=('quantile',),
        attrs={'scale': scale, 'sentinel': sentinel, 'offset': offset})


def test_decode_applies_scale_and_sentinel():
    out = q.decode(_da([0, 100, -32768, 32767], 3.0518509e-05, -32768))
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(100 * 3.0518509e-05, rel=1e-6)
    assert np.isnan(out[2])
    assert out[3] == pytest.approx(1.0, abs=1e-4)


def test_decode_reads_the_scale_from_the_array_not_a_constant():
    """A re-export with a different scale must follow the file, not the code."""
    raw = [1000]
    a = q.decode(_da(raw, 3.0518509e-05, -32768))
    b = q.decode(_da(raw, 1.0e-04, -32768))
    assert not np.isclose(a[0], b[0])
    assert b[0] == pytest.approx(0.1)


def test_decode_honours_a_different_sentinel():
    out = q.decode(_da([-9999, 5], 1e-04, -9999))
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(5e-04)


def test_decode_applies_the_offset():
    out = q.decode(_da([10], 0.1, -32768, offset=2.0))
    assert out[0] == pytest.approx(3.0)


# --- the quantile function as a distribution ---------------------------------

P = np.linspace(0.001, 0.999, 64)


def _linear_qf(lo=0.0, hi=1.0):
    """A strictly increasing QF: the uniform distribution on [lo, hi]."""
    return (lo + P * (hi - lo))[:, None]


def _atom_qf(mass=0.3):
    """A QF with an atom at zero: flat for the first `mass` of the levels."""
    return np.where(P < mass, 0.0, (P - mass) / (1.0 - mass))[:, None]


def test_cdf_is_the_inverse_of_a_strictly_increasing_qf():
    v = _linear_qf()
    for y in (0.1, 0.25, 0.5, 0.9):
        assert q.cdf_at(P, v, np.array([y]))[0] == pytest.approx(y, abs=0.01)


def test_cdf_saturates_outside_the_support():
    v = _linear_qf(0.2, 0.8)
    assert q.cdf_at(P, v, np.array([0.0]))[0] == 0.0
    assert q.cdf_at(P, v, np.array([1.0]))[0] == 1.0


def test_bracket_straddles_an_atom():
    """On a plateau F- and F+ differ; that gap IS the probability mass there."""
    v = _atom_qf(mass=0.3)
    f_minus, f_plus = q._bracket(P, v, np.array([0.0]))
    assert f_minus[0] < 0.01
    assert f_plus[0] == pytest.approx(0.3, abs=0.02)


def test_value_on_the_lowest_quantile_is_on_the_support_not_under_it():
    """The regression this module's boundary logic exists for.

    Testing "no levels strictly below y" cannot distinguish y sitting ON the
    lowest value from y sitting under the whole distribution. With HM's atom at
    no-change that mistake collapses ~30% of pixels onto PIT = 0 and a
    perfectly calibrated forecast reads as wildly over-confident.
    """
    v = _atom_qf(mass=0.3)
    f_minus, f_plus = q._bracket(P, v, np.array([0.0]))
    assert f_plus[0] > f_minus[0], 'the atom must carry mass'

    below = q._bracket(P, v, np.array([-0.5]))
    assert below[0][0] == 0.0 and below[1][0] == 0.0


def test_pit_is_uniform_for_a_calibrated_forecast():
    """Draw the truth from the predicted distribution; the PIT must be flat."""
    rng = np.random.default_rng(0)
    n = 20_000
    mu = rng.uniform(0.1, 0.6, n)
    sigma = rng.uniform(0.02, 0.15, n)

    from scipy.stats import norm
    v = mu[None, :] + sigma[None, :] * norm.ppf(P)[:, None]
    y = rng.normal(mu, sigma)

    u = q.pit(P, v, y, rng)
    counts, _ = np.histogram(u, bins=10, range=(0, 1))
    expected = n / 10
    # +/-6% of the expected bin height; the 64-knot grid adds a little noise
    assert np.all(np.abs(counts - expected) < 0.06 * expected), counts


def test_pit_detects_an_overconfident_forecast():
    """Halving the predicted spread must push the PIT into a U shape."""
    rng = np.random.default_rng(1)
    n = 20_000
    mu = np.full(n, 0.4)
    from scipy.stats import norm
    v = mu[None, :] + 0.05 * norm.ppf(P)[:, None]     # claimed spread
    y = rng.normal(mu, 0.15)                          # true spread, 3x wider

    u = q.pit(P, v, y, rng)
    counts, _ = np.histogram(u, bins=10, range=(0, 1))
    edges = counts[0] + counts[-1]
    middle = counts[4] + counts[5]
    assert edges > 3 * middle, (edges, middle)


def test_pit_randomisation_spreads_the_atom():
    """Without randomisation every atom pixel would return the same PIT."""
    rng = np.random.default_rng(2)
    n = 5000
    v = np.repeat(_atom_qf(mass=0.3), n, axis=1)
    y = np.zeros(n)
    u = q.pit(P, v, y, rng)
    assert u.min() < 0.05 and u.max() > 0.25
    assert np.unique(np.round(u, 4)).size > 100


def test_bin_probs_sum_to_one_across_the_full_range():
    v = _linear_qf(0.1, 0.9)
    probs = q.bin_probs(P, v, np.array([-1.0, 0.3, 0.5, 0.7, 2.0]))
    assert probs.shape == (4, 1)
    assert probs.sum(axis=0)[0] == pytest.approx(1.0, abs=1e-9)
    assert np.all(probs >= 0)


def test_bin_probs_accepts_per_column_edges():
    """Fig 5 shifts each pixel's edges by its own 2000 observation."""
    v = np.column_stack([_linear_qf(0.0, 1.0), _linear_qf(0.5, 1.5)])
    edges = np.array([[-1.0, -0.5], [0.5, 1.0], [3.0, 3.0]])
    probs = q.bin_probs(P, v, edges)
    assert probs.shape == (2, 2)
    assert probs.sum(axis=0) == pytest.approx([1.0, 1.0], abs=1e-9)
    # both columns are cut at their own midpoint, so both split ~50/50
    assert probs[0] == pytest.approx([0.5, 0.5], abs=0.02)


def test_density_integrates_to_one():
    v = _linear_qf(0.2, 0.8)[:, 0]
    grid = np.linspace(0.0, 1.0, 2001)
    d = q.density(P, v, grid)
    assert np.trapezoid(d, grid) == pytest.approx(1.0, abs=0.02)
    assert np.all(d >= 0)


def test_density_of_a_uniform_is_flat_inside_the_support():
    v = _linear_qf(0.25, 0.75)[:, 0]
    grid = np.linspace(0.0, 1.0, 1001)
    d = q.density(P, v, grid)
    inside = (grid > 0.3) & (grid < 0.7)
    assert d[inside].std() < 0.15 * d[inside].mean()
    assert d[grid < 0.2].max() < 1e-6


# --- the fan --------------------------------------------------------------


def test_anchor_prepends_the_last_observation():
    t = np.array([2005.0, 2010.0])
    v = np.array([[0.1, 0.2], [0.3, 0.5]])          # 2 levels x 2 times
    obs_t = np.array([1990.0, 1995.0, 2000.0, 2005.0])
    obs_v = np.array([0.02, 0.03, 0.05, 0.06])

    t2, v2 = q.anchor_to_observation(t, v, obs_t, obs_v)
    assert t2[0] == 2000.0                           # last obs before 2005
    assert v2.shape == (2, 3)
    assert np.allclose(v2[:, 0], 0.05)               # fan collapses to a point


def test_anchor_is_a_no_op_without_a_usable_observation():
    t = np.array([2005.0])
    v = np.array([[0.1], [0.3]])
    t2, v2 = q.anchor_to_observation(t, v, np.array([2010.0]), np.array([0.5]))
    assert np.array_equal(t2, t) and np.array_equal(v2, v)


def test_anchor_skips_a_missing_observation():
    t = np.array([2005.0])
    v = np.array([[0.1], [0.3]])
    t2, v2 = q.anchor_to_observation(
        t, v, np.array([2000.0]), np.array([np.nan]))
    assert np.array_equal(t2, t) and np.array_equal(v2, v)


def test_fan_image_shape_and_range():
    t = np.array([2005.0, 2010.0, 2015.0, 2020.0])
    v = np.linspace(0.1, 0.6, 64)[:, None] + np.zeros((1, 4))
    t_f, y, A, M = q.fan_image(t, P, v, ny=64, nt=32)
    assert A.shape == (64, 32) and M.shape == (64, 32)
    assert A.min() >= 0.0 and A.max() <= 1.0
    assert M.any()


def test_fan_image_is_brightest_at_the_median():
    t = np.array([2005.0, 2020.0])
    from scipy.stats import norm
    v = 0.4 + 0.1 * norm.ppf(P)[:, None] + np.zeros((1, 2))
    _, y, A, M = q.fan_image(t, P, v, ny=201, nt=8, mode='prob')
    col = A[:, 4]
    assert y[int(np.argmax(col))] == pytest.approx(0.4, abs=0.03)
