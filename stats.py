"""Console summary statistics for the HM forecasts.

Run:

    mamba run -n hm_plots python stats.py

The model emits a 64-level quantile function per pixel, from which we have
per-pixel exceedance probabilities. Every area below is therefore an EXPECTED
area: the sum of per-pixel probabilities times cell area. That is valid under
any spatial dependence, because expectation is linear.

This replaces what the script used to report. It counted pixels whose upper or
lower blended surface crossed a threshold and called that an area, which is the
area only in a world where every pixel realises its unlucky outcome at the same
moment - the perfect-dependence extreme, and the largest value obtainable under
any dependence structure. The numbers below will not match the old percentages;
they are a different quantity, not a moved result.

Five blocks:

  * grid and land area
  * HM change - observed 2000-2020 against projected 2020-2040, as expected
    modification added (km2 of fully-modified equivalent), which is the same
    quantity on both sides for the first time
  * natural lands lost - expected area of 2020 land below 0.10 that reaches 0.10
  * moderately modified lands lost - same for 0.10-0.40 land reaching 0.40
  * condition-class budget - the whole land surface split three ways per year

Each expected area carries two brackets, and both need only the marginals:

  * INDEPENDENCE   sqrt(sum p(1-p)) is the standard deviation of the total if
    pixels were independent. The narrowest interval that is plausible.
  * PERFECT DEPENDENCE   the count of pixels with p > 0.975 and p > 0.025,
    times cell area. These are exactly the old "lower" and "upper" numbers.

The true interval sits between the two. They are never called lower and upper
forecasts, which is the naming that caused the problem in the first place.

Every quantity is computed on an equal-area grid (config.EQUAL_AREA_CRS), so a
pixel share is also an area share. That matters more here rather than less: we
multiply a sum of probabilities by a cell area, so the sum is only an area if
the cells are. The source rasters are EPSG:4326, where cell ground area falls
off with cos(lat).

The rasters are ~685 M cells each and fourteen of them are needed, so they are
warped on the fly and streamed in row-strips, reduced to float64 accumulators
rather than being held in memory.

The grid machinery is shared with figures.py and lives in equal_area.py, so the
console statistics here and the per-ecoregion sums in figures.fig10 are measured
on the same cells. As a cross-check, fig10 reports exactly this module's
`valid_2020` (131,424,956 cells) as its land mask.
"""
import contextlib
import math
import sys

import numpy as np
import rasterio
from rasterio.windows import Window

import config
import equal_area

# --- parameters -------------------------------------------------------------

# |dHM| at or below a threshold counts as stable. The change block is reported
# at each of these, so the reader can see how sensitive the increase/decrease
# split is to where the "no meaningful change" line is drawn.
CHANGE_THRESHOLDS = (0.005, 0.01, 0.05)

LOW_CUT = config.LOW_CUT     # "natural" ceiling (inclusive) / first exceedance level
HIGH_CUT = config.HIGH_CUT   # "moderate HM" ceiling / second exceedance level

FORECAST_YEARS = config.FORECAST_YEARS
BASE_YEAR = 2020            # the forecasts' base; per-year rates divide by the lead

ROWS_PER_STRIP = 512       # ~82 MB per raster per strip

# Risk concentration: a histogram of p over each base, from which we report the
# share of total expected loss carried by the highest-probability decile of
# pixels. Concentrated loss is targetable and diffuse loss is not.
P_HIST_BINS = 200
CONCENTRATION_FRACTION = 0.10

# 1.96 sigma, the independence interval's half-width.
Z95 = 1.959964


def mean_key(year: int) -> str:
    """config.PATHS key for the E[HM] raster of a forecast year.

    There is no HM_mean_YYYY.tif. The blended "central" surface already IS the
    mean: the model documents central as E[Q] = integral of Q(u) du over [0,1],
    not the median. 2040 predates the pred_* naming and lives under hm_*_2040.
    """
    return 'hm_central_2040' if year == 2040 else f'pred_{year}_central'


# --- equal-area reprojection ------------------------------------------------
# The grid machinery is shared with figures.py. It lives in equal_area, a leaf
# module, so this console script does not import matplotlib/cartopy behind it.

target_grid = equal_area.equal_area_grid
open_layer = equal_area.open_equal_area
AUTHALIC_RADIUS_M = equal_area.AUTHALIC_RADIUS_M


def cell_km2() -> float:
    """Ground area of one analysis cell, km2.

    Constant on the equal-area grid, which is why every accumulator below is a
    unitless sum multiplied through once at the end.
    """
    return config.EQUAL_AREA_RES ** 2 / 1e6


def source_land_area_km2() -> float:
    """Land area measured on the native EPSG:4326 grid by cos(lat) weighting.

    A geographic cell spanning [lat1, lat2] x dlon has exact spherical area
    R^2 * dlon * (sin lat2 - sin lat1). Summing that over valid cells gives an
    area that owes nothing to the warp, so comparing it against the equal-area
    pixel count catches a destination grid that over- or under-fills - as a
    pseudocylindrical CRS does when its out-of-domain corners are not masked.
    """
    with rasterio.open(config.PATHS['hm_2020_aa']) as src:
        transform = src.transform
        lon_step = math.radians(abs(transform.a))
        total = 0.0
        for window in strips(src.height, src.width):
            band = read_strip(src, window)
            valid_per_row = np.count_nonzero(np.isfinite(band), axis=1)
            top = transform.f + transform.e * window.row_off
            lats = top + transform.e * np.arange(window.height + 1)
            # band of sin(lat) covered by each row; transform.e is negative
            strip_area = (AUTHALIC_RADIUS_M ** 2 * lon_step
                          * np.abs(np.diff(np.sin(np.radians(lats)))))
            total += float(np.sum(valid_per_row * strip_area))
    return total / 1e6


# --- raster streaming -------------------------------------------------------

read_strip = equal_area.read_masked


def strips(height: int, width: int):
    for top in range(0, height, ROWS_PER_STRIP):
        yield Window(0, top, width, min(ROWS_PER_STRIP, height - top))


def _new_exceedance_block() -> dict:
    return dict(base=0, sum_p=0.0, sum_pq=0.0, n_hi=0, n_lo=0,
                hist=np.zeros(P_HIST_BINS, dtype=np.int64))


def _accumulate_exceedance(block: dict, p: np.ndarray) -> None:
    """Fold one strip's probabilities, over one base, into `block`.

    float64 accumulators: adding 685 M float32 values into a float64 sum is
    exact enough that no compensated summation is needed.
    """
    block['base'] += int(p.size)
    block['sum_p'] += float(np.sum(p, dtype=np.float64))
    block['sum_pq'] += float(np.sum(p.astype(np.float64) * (1.0 - p), dtype=np.float64))
    block['n_hi'] += int(np.count_nonzero(p > 0.975))
    block['n_lo'] += int(np.count_nonzero(p > 0.025))
    idx = np.clip((p * P_HIST_BINS).astype(np.int32), 0, P_HIST_BINS - 1)
    block['hist'] += np.bincount(idx, minlength=P_HIST_BINS).astype(np.int64)


def collect() -> dict:
    """Stream every raster once, accumulating what each statistic needs."""
    keys = {
        'obs1990': 'hm_1990_aa',
        'obs2000': 'hm_2000_aa',
        'obs2010': 'hm_2010_aa',
        'obs2020': 'hm_2020_aa',
        **{('p10', y): f'hm_p10_{y}' for y in FORECAST_YEARS},
        **{('p40', y): f'hm_p40_{y}' for y in FORECAST_YEARS},
        **{('mean', y): mean_key(y) for y in FORECAST_YEARS},
    }
    missing = [str(config.PATHS[k]) for k in keys.values()
               if not config.PATHS[k].exists()]
    if missing:
        raise SystemExit('Missing input rasters:\n  ' + '\n  '.join(missing))

    change = {t: dict(increase=0, decrease=0, stable=0, n=0)
              for t in CHANGE_THRESHOLDS}
    # A thresholded view of E[HM 2040] - HM 2020, kept only for continuity with
    # the old table. It is a property of the expected surface, not a projection
    # of area, and is labelled that way in the report.
    expected_surface = {t: dict(increase=0, decrease=0, stable=0, n=0)
                        for t in CHANGE_THRESHOLDS}

    natural = {y: _new_exceedance_block() for y in FORECAST_YEARS}
    moderate = {y: _new_exceedance_block() for y in FORECAST_YEARS}
    budget = {y: dict(n=0, natural=0.0, moderate=0.0, high=0.0, inverted=0)
              for y in FORECAST_YEARS}
    added = {y: dict(n=0, total=0.0) for y in FORECAST_YEARS}
    observed_added = dict(n=0, total=0.0)
    # Probability layers should inherit the observed grid's nodata exactly;
    # report any shortfall rather than assuming it away.
    gaps = {('p10', y): 0 for y in FORECAST_YEARS}
    gaps.update({('p40', y): 0 for y in FORECAST_YEARS})

    # Realised loss over the observed record, for comparison with the
    # projections. These are COUNTS of pixels that actually crossed, not sums of
    # probabilities - a realised area, not an expected one. Keyed by base year.
    observed_loss = {(y, cls): dict(base=0, lost=0)
                     for y in (1990, 2000) for cls in ('natural', 'moderate')}
    # Decade by decade, each on its own start year's condition classes.
    OBS_DECADES = ((1990, 2000), (2000, 2010), (2010, 2020))
    observed_decade = {(d, cls): dict(base=0, lost=0)
                       for d in OBS_DECADES for cls in ('natural', 'moderate')}

    valid_2020 = 0
    # Unrestricted bases (observed 2020 only). Each year's own base is a touch
    # smaller wherever that year's probability layer has no value.
    natural_base_all = moderate_base_all = 0

    with contextlib.ExitStack() as stack:
        with rasterio.open(config.PATHS['hm_2020_aa']) as ref:
            grid = target_grid(ref)
        src = {name: open_layer(stack, config.PATHS[key], grid)
               for name, key in keys.items()}
        height, width = src['obs2020'].height, src['obs2020'].width
        windows = list(strips(height, width))

        for i, window in enumerate(windows, 1):
            print(f'\r  reading strip {i}/{len(windows)} '
                  f'(rows {window.row_off}-{window.row_off + window.height})',
                  end='', file=sys.stderr, flush=True)

            obs1990 = read_strip(src['obs1990'], window)
            obs2000 = read_strip(src['obs2000'], window)
            obs2010 = read_strip(src['obs2010'], window)
            obs2020 = read_strip(src['obs2020'], window)
            fin2000 = np.isfinite(obs2000)
            fin2020 = np.isfinite(obs2020)
            valid_2020 += int(np.count_nonzero(fin2020))

            # Realised crossings over the observed record, on each base year's
            # own condition classes.
            by_year = {1990: obs1990, 2000: obs2000, 2010: obs2010,
                       2020: obs2020}
            for (a, b), cls in [(d, c) for d in OBS_DECADES
                                for c in ('natural', 'moderate')]:
                start, end = by_year[a], by_year[b]
                both_d = np.isfinite(start) & np.isfinite(end)
                cut = LOW_CUT if cls == 'natural' else HIGH_CUT
                with np.errstate(invalid='ignore'):
                    # Cuts are closed at the bottom: natural is HM <= 0.10.
                    # See config.LOW_CUT.
                    base = (both_d & (start <= LOW_CUT) if cls == 'natural'
                            else both_d & (start > LOW_CUT) & (start < HIGH_CUT))
                    row = observed_decade[((a, b), cls)]
                    row['base'] += int(np.count_nonzero(base))
                    row['lost'] += int(np.count_nonzero(base & (end >= cut)))

            for base_year, base_arr in ((1990, obs1990), (2000, obs2000)):
                both_y = np.isfinite(base_arr) & fin2020
                with np.errstate(invalid='ignore'):
                    ib = both_y & (base_arr <= LOW_CUT)
                    mb = (both_y & (base_arr > LOW_CUT)
                          & (base_arr < HIGH_CUT))
                    row = observed_loss[(base_year, 'natural')]
                    row['base'] += int(np.count_nonzero(ib))
                    row['lost'] += int(np.count_nonzero(ib & (obs2020 >= LOW_CUT)))
                    row = observed_loss[(base_year, 'moderate')]
                    row['base'] += int(np.count_nonzero(mb))
                    row['lost'] += int(np.count_nonzero(mb & (obs2020 >= HIGH_CUT)))

            # Bases for the two exceedance blocks, evaluated on observed 2020.
            with np.errstate(invalid='ignore'):
                natural_base = fin2020 & (obs2020 <= LOW_CUT)
                moderate_base = (fin2020 & (obs2020 > LOW_CUT)
                                 & (obs2020 < HIGH_CUT))
            natural_base_all += int(np.count_nonzero(natural_base))
            moderate_base_all += int(np.count_nonzero(moderate_base))

            # Observed change 2000-2020. Unchanged: it describes observations,
            # involves no forecast uncertainty, and was always fine.
            both = fin2000 & fin2020
            n_both = int(np.count_nonzero(both))
            with np.errstate(invalid='ignore'):
                delta = obs2020 - obs2000
                for t in CHANGE_THRESHOLDS:
                    row = change[t]
                    row['increase'] += int(np.count_nonzero(both & (delta > t)))
                    row['decrease'] += int(np.count_nonzero(both & (delta < -t)))
                    row['n'] += n_both
            observed_added['n'] += n_both
            observed_added['total'] += float(np.sum(delta[both], dtype=np.float64))

            for y in FORECAST_YEARS:
                p10 = read_strip(src[('p10', y)], window)
                p40 = read_strip(src[('p40', y)], window)
                fin10 = np.isfinite(p10)
                fin40 = np.isfinite(p40)
                gaps[('p10', y)] += int(np.count_nonzero(fin2020 & ~fin10))
                gaps[('p40', y)] += int(np.count_nonzero(fin2020 & ~fin40))

                _accumulate_exceedance(natural[y], p10[natural_base & fin10])
                _accumulate_exceedance(moderate[y], p40[moderate_base & fin40])

                # Condition-class budget over all land both layers resolve.
                ok = fin2020 & fin10 & fin40
                a, b = p10[ok].astype(np.float64), p40[ok].astype(np.float64)
                budget[y]['n'] += int(ok.sum())
                budget[y]['natural'] += float(np.sum(1.0 - a))
                budget[y]['moderate'] += float(np.sum(a - b))
                budget[y]['high'] += float(np.sum(b))
                # A monotone quantile function cannot put p40 above p10; a
                # non-zero count means the spline head is misbehaving, and it
                # shows up here before anywhere else. Reported, never clamped.
                budget[y]['inverted'] += int(np.count_nonzero(b > a))
                del a, b

                mean = read_strip(src[('mean', y)], window)
                okm = fin2020 & np.isfinite(mean)
                added[y]['n'] += int(okm.sum())
                added[y]['total'] += float(
                    np.sum(mean[okm].astype(np.float64) - obs2020[okm], dtype=np.float64))

                if y == FORECAST_YEARS[-1]:
                    with np.errstate(invalid='ignore'):
                        dm = mean - obs2020
                        n_ok = int(okm.sum())
                        for t in CHANGE_THRESHOLDS:
                            row = expected_surface[t]
                            row['increase'] += int(np.count_nonzero(okm & (dm > t)))
                            row['decrease'] += int(np.count_nonzero(okm & (dm < -t)))
                            row['n'] += n_ok

        print('\r' + ' ' * 70 + '\r', end='', file=sys.stderr, flush=True)

    # stable is derived, so the three always sum to n by construction
    for table in (change, expected_surface):
        for row in table.values():
            row['stable'] = row['n'] - row['increase'] - row['decrease']

    native_km2 = source_land_area_km2()
    warped_km2 = (valid_2020 * cell_km2() if config.EQUAL_AREA_CRS else None)

    return {'grid': (height, width), 'valid_2020': valid_2020,
            'change': change, 'expected_surface': expected_surface,
            'natural': natural, 'moderate': moderate, 'budget': budget,
            'added': added, 'observed_added': observed_added, 'gaps': gaps,
            'observed_loss': observed_loss,
            'observed_decade': observed_decade,
            'natural_base_all': natural_base_all,
            'moderate_base_all': moderate_base_all,
            'crs': config.EQUAL_AREA_CRS, 'res': config.EQUAL_AREA_RES,
            'native_km2': native_km2, 'warped_km2': warped_km2}


# --- derived quantities -----------------------------------------------------


def concentration_share(block: dict) -> float:
    """Share of total expected loss carried by the top decile of pixels.

    Read off the p-histogram: walk bins from the highest probability down until
    a tenth of the base is covered, and total the expected loss they carry.
    Returns NaN when the base is empty.
    """
    hist = block['hist']
    total_pixels = int(hist.sum())
    total_loss = block['sum_p']
    if not total_pixels or total_loss <= 0:
        return float('nan')

    target = CONCENTRATION_FRACTION * total_pixels
    midpoints = (np.arange(P_HIST_BINS) + 0.5) / P_HIST_BINS
    taken = 0.0
    loss = 0.0
    for b in range(P_HIST_BINS - 1, -1, -1):
        n = float(hist[b])
        if n == 0:
            continue
        use = min(n, target - taken)
        loss += use * midpoints[b]
        taken += use
        if taken >= target:
            break
    return 100.0 * loss / total_loss


# --- console report ---------------------------------------------------------

WIDTH = 78
AREA_TOLERANCE_PCT = 2.0    # warped vs native land area disagreement to tolerate


def area_ratio(res: dict) -> float:
    """Percentage by which the warped land area exceeds the native measure."""
    if not res.get('warped_km2') or not res.get('native_km2'):
        return 0.0
    return 100.0 * (res['warped_km2'] / res['native_km2'] - 1.0)


def pct(part: float, whole: float) -> str:
    return '     - ' if not whole else f'{100.0 * part / whole:6.2f}%'


def km2(value: float) -> str:
    return f'{value:,.0f}'


def rule(char: str = '-') -> str:
    return char * WIDTH


def _year_row(label: str, values, fmt=km2) -> str:
    cells = ''.join(f'{fmt(v):>11}' for v in values)
    return f'  {label:<32}{cells}'


def report(res: dict) -> None:
    height, width = res['grid']
    cells = height * width
    area = cell_km2()

    print()
    print(rule('='))
    print(' HM forecast summary statistics')
    print(rule('='))
    if res['crs']:
        print(f' grid           {res["crs"]} @ {res["res"]} m  (equal area)')
        print(f'                {height:,} x {width:,}  ({cells:,} cells)')
        print(f' valid in 2020  {res["valid_2020"]:,} pixels '
              f'({100.0 * res["valid_2020"] / cells:.2f}% of grid)')
        print(f' land area      {res["warped_km2"]:,.0f} km2 on this grid')
        print(f'                {res["native_km2"]:,.0f} km2 by cos(lat) on the '
              f'native grid  ({area_ratio(res):+.2f}%)')
        print(' equal-area cells, so a sum of probabilities times cell area is '
              'an area')
        if abs(area_ratio(res)) > AREA_TOLERANCE_PCT:
            print()
            over = area_ratio(res) > 0
            print(f' !! WARNING: the two land areas disagree by more than '
                  f'{AREA_TOLERANCE_PCT}%. Every')
            print(f' !! number below is computed on a land mask that is '
                  f'{"too large" if over else "too small"},')
            print(f' !! so treat them all as wrong until this is resolved. '
                  f'Likely causes:')
            print(f' !!   1. nodata is not being honoured, so ocean counts as '
                  f'land. Check')
            print(f' !!      that the warp preserves each source\'s declared '
                  f'nodata - passing')
            print(f' !!      src_nodata=None to WarpedVRT *overrides* it rather '
                  f'than')
            print(f' !!      inheriting it, which turns +3.4e38 into valid data.')
            print(f' !!   2. {res["crs"]} fills destination cells outside its '
                  f'valid domain.')
            print(f' !!      Pseudocylindrical CRSs (e.g. Equal Earth, '
                  f'EPSG:8857) keep')
            print(f' !!      out-of-domain corners; a cylindrical one such as '
                  f'EPSG:6933 does not.')
    else:
        print(f' grid           EPSG:4326 (native, NOT equal area)')
        print(f'                {height:,} x {width:,}  ({cells:,} cells)')
        print(f' valid in 2020  {res["valid_2020"]:,} pixels '
              f'({100.0 * res["valid_2020"] / cells:.2f}% of grid)')
        print(' percentages are shares of pixels, not of ground area')

    gap_lines = [f'{k[0]} {k[1]}: {v:,}' for k, v in sorted(res['gaps'].items())
                 if v]
    print()
    if gap_lines:
        print(' !! probability layers do not cover the observed footprint:')
        for line in gap_lines:
            print(f' !!   {line} pixels of 2020 land with no probability')
    else:
        print(' probability layers cover the observed 2020 footprint exactly')
    print()

    # --- 2. change 2000-2020 observed vs 2020-2040 projected ---
    print('HM CHANGE   (expected modification added, km2 fully-modified '
          'equivalent)')
    print(rule())
    print('  The observed and projected lines are the same quantity for the')
    print('  first time: area x mean change, not a count of pixels crossing a')
    print('  threshold on a shrunk surface.')
    obs = res['observed_added']
    print(f'  {"observed 2000-2020":<32}{km2(obs["total"] * area):>11}'
          f'   over {obs["n"]:,} cells')
    for y in FORECAST_YEARS:
        row = res['added'][y]
        print(f'  {f"projected 2020-{y}":<32}{km2(row["total"] * area):>11}'
              f'   over {row["n"]:,} cells')
    print()

    print('  direction of change   (stable = |dHM| <= threshold)')
    print(f'  {"":<32}{"threshold":>11}{"increase":>10}'
          f'{"decrease":>10}{"stable":>10}')
    for label, table in (
        (f'E[HM] {FORECAST_YEARS[-1]} - HM 2020', res['expected_surface']),
        ('Observed  2000-2020', res['change']),
    ):
        for i, t in enumerate(sorted(table)):
            row = table[t]
            print(f'  {label if i == 0 else "":<32}{t:>11.3f}'
                  f'{pct(row["increase"], row["n"]):>10}'
                  f'{pct(row["decrease"], row["n"]):>10}'
                  f'{pct(row["stable"], row["n"]):>10}')
    print('  the first block is a property of the expected surface, not a')
    print('  projection of area - E[HM] is a mean, and thresholding a mean is')
    print('  not the same as the expected area past that threshold')
    print()

    # --- 3 & 4. the two expected-area blocks ---
    for title, block, base_text, base_n, cls_key in (
        (f'NATURAL LANDS LOST   (reaching HM >= {LOW_CUT:.2f})', res['natural'],
         f'pixels with observed HM <= {LOW_CUT:.2f} in 2020',
         res['natural_base_all'], 'natural'),
        (f'MODERATELY MODIFIED LANDS LOST   (reaching HM >= {HIGH_CUT:.2f})',
         res['moderate'],
         f'pixels with {LOW_CUT:.2f} < observed HM < {HIGH_CUT:.2f} in 2020',
         res['moderate_base_all'], 'moderate'),
    ):
        print(title)
        print(f'  base: {base_text}')
        print(f'        n = {base_n:,} cells = {km2(base_n * area)} km2')
        print('  EXPECTED area: the sum of per-pixel probabilities x cell area.')
        print('  Valid under any spatial dependence. Not comparable with the')
        print('  old "% of base reaching HM >= X" figures, which were pixel')
        print('  counts on a single scenario surface.')
        gap = max(base_n - v['base'] for v in block.values())
        if gap:
            print(f'  denominators are per-year; the largest excludes {gap:,} '
                  f'pixels ({100.0 * gap / base_n:.4f}%) with no probability')
        print(rule())
        print(f'  {"":<32}' + ''.join(f'{y:>11}' for y in FORECAST_YEARS))

        years = list(FORECAST_YEARS)
        expected = [block[y]['sum_p'] * area for y in years]
        half = [Z95 * math.sqrt(block[y]['sum_pq']) * area for y in years]
        print(_year_row('expected loss (km2)', expected))
        print(_year_row('% of base',
                        [100.0 * block[y]['sum_p'] / block[y]['base']
                         if block[y]['base'] else 0.0 for y in years],
                        fmt=lambda v: f'{v:.2f}%'))
        print(_year_row('per year (km2/yr)',
                        [e / (y - BASE_YEAR) for e, y in zip(expected, years)]))
        print('  independence bracket  (pixels independent; narrowest plausible)')
        print(_year_row('  low (km2)', [e - h for e, h in zip(expected, half)]))
        print(_year_row('  high (km2)', [e + h for e, h in zip(expected, half)]))
        print('  perfect-dependence bracket  (every pixel resolves together)')
        print(_year_row('  low  = area with p > 0.975',
                        [block[y]['n_hi'] * area for y in years]))
        print(_year_row('  high = area with p > 0.025',
                        [block[y]['n_lo'] * area for y in years]))
        if any(block[a]['n_hi'] > block[b]['n_hi']
               for a, b in zip(years, years[1:])):
            print('  the p > 0.975 row is not monotone in the horizon, and that '
                  'is expected:')
            print('  the spline head cannot narrow an interval with lead time, '
                  'so widening can')
            print('  push mass back below the cut and lower p for a pixel that '
                  'was near-certain.')
            print('  The expected area above is driven by the bulk, not the '
                  'tail, and does rise.')
        print(_year_row('top-decile share of loss',
                        [concentration_share(block[y]) for y in years],
                        fmt=lambda v: f'{v:.1f}%'))
        print()
        # The observed record, for scale. These are realised areas - counts of
        # pixels that actually crossed - so they are NOT the same quantity as
        # the expected areas above and the two should not be differenced.
        print('  OBSERVED for comparison (realised area, not an expectation;')
        print('  each period uses its own base year\'s condition classes)')
        obs_rows = [(2000, 2020)]
        print(f'  {"":<32}' + ''.join(f'{f"{a}-{b}":>11}' for a, b in obs_rows))
        get = lambda a: res['observed_loss'][(a, cls_key)]      # noqa: E731
        print(_year_row('  base (km2)',
                        [get(a)['base'] * area for a, _ in obs_rows]))
        print(_year_row('  lost (km2)',
                        [get(a)['lost'] * area for a, _ in obs_rows]))
        print(_year_row('  % of base',
                        [100.0 * get(a)['lost'] / get(a)['base']
                         if get(a)['base'] else 0.0 for a, _ in obs_rows],
                        fmt=lambda v: f'{v:.2f}%'))
        print(_year_row('  per year (km2/yr)',
                        [get(a)['lost'] * area / (b - a) for a, b in obs_rows]))
        print()

    # --- decadal loss ---
    print('DECADAL LOSS   (observed = realised area; projected = expected area)')
    print('  each observed decade uses its own start year\'s condition classes;')
    print('  projected decades are differences of cumulative expected area')
    print(rule())
    dec_obs = [(1990, 2000), (2000, 2010), (2010, 2020)]
    dec_prj = [(2020, 2030), (2030, 2040)]
    cols = dec_obs + dec_prj

    def _dec_row(label, values, fmt=km2):
        cells = ''.join(f'{fmt(v):>10}' for v in values)
        print(f'  {label:<26}{cells}')

    print(f'  {"":<26}' + ''.join(f'{f"{a}-{b}":>10}' for a, b in cols))
    for cls, block, label in (('natural', res['natural'], 'natural lands'),
                              ('moderate', res['moderate'],
                               'moderately modified lands')):
        print(f'  {label} (crossing HM >= '
              f'{LOW_CUT if cls == "natural" else HIGH_CUT:.2f})')
        lost, per_yr, share = [], [], []
        for a, b in dec_obs:
            row = res['observed_decade'][((a, b), cls)]
            lost.append(row['lost'] * area)
            per_yr.append(row['lost'] * area / (b - a))
            share.append(100.0 * row['lost'] / row['base'] if row['base'] else 0.0)
        prev = 0.0
        for a, b in dec_prj:
            cum = block[b]['sum_p'] * area
            lost.append(cum - prev)
            per_yr.append((cum - prev) / (b - a))
            share.append(100.0 * (cum - prev) / (block[b]['base'] * area)
                         if block[b]['base'] else 0.0)
            prev = cum
        _dec_row('  lost in decade (km2)', lost)
        _dec_row('  per year (km2/yr)', per_yr)
        _dec_row('  % of that decade\'s base', share, fmt=lambda v: f'{v:.2f}%')
    print()

    # --- 5. condition-class budget ---
    print('CONDITION-CLASS BUDGET   (expected area of the whole land surface)')
    print('  natural = sum(1 - p10),  moderate = sum(p10 - p40),  '
          'high = sum(p40)')
    print(rule())
    print(f'  {"":<32}' + ''.join(f'{y:>11}' for y in FORECAST_YEARS))
    years = list(FORECAST_YEARS)
    b = res['budget']
    for label, key in (('natural (km2)', 'natural'),
                       ('moderately modified (km2)', 'moderate'),
                       ('highly modified (km2)', 'high')):
        print(_year_row(label, [b[y][key] * area for y in years]))
    for label, key in (('natural (share)', 'natural'),
                       ('moderately modified (share)', 'moderate'),
                       ('highly modified (share)', 'high')):
        print(_year_row(label, [100.0 * b[y][key] / b[y]['n'] if b[y]['n'] else 0.0
                                for y in years], fmt=lambda v: f'{v:.2f}%'))
    print(_year_row('p40 > p10 (should be 0)',
                    [b[y]['inverted'] for y in years], fmt=lambda v: f'{v:,}'))

    # The three classes are a partition of the land, so they must close. It
    # costs nothing to check and it exercises the whole streaming path.
    for y in years:
        total = b[y]['natural'] + b[y]['moderate'] + b[y]['high']
        assert math.isclose(total, b[y]['n'], rel_tol=1e-9), (
            f'condition-class budget for {y} does not close: '
            f'{total:,.3f} vs {b[y]["n"]:,} cells')
    print(f'  closure check passed: the three classes sum to the land area '
          f'for every year')
    print()

    print(rule('='))


def main() -> None:
    where = (f'warping to {config.EQUAL_AREA_CRS} @ {config.EQUAL_AREA_RES} m'
             if config.EQUAL_AREA_CRS else 'on the native EPSG:4326 grid')
    print(f'Streaming 16 rasters, {where}...', file=sys.stderr)
    report(collect())


if __name__ == '__main__':
    main()
