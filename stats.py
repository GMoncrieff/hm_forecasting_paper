"""Console summary statistics for the HM forecasts.

Run:

    mamba run -n hm_plots python stats.py

Prints three blocks of percentages:

  * direction of HM change - observed 2000-2020 against projected 2020-2040
  * share of low-HM (< 0.10) land in 2020 that reaches HM >= 0.10 by 2025-2040
  * share of moderate-HM (0.10-0.40) land in 2020 that reaches HM >= 0.40

Every percentage is a share of pixels counted on an equal-area grid
(config.EQUAL_AREA_CRS), so a pixel share is also an area share. The source
rasters are EPSG:4326, where cell ground area falls off with cos(lat); counting
on that grid would over-weight high latitudes.

The rasters are ~685 M cells each and fourteen of them are needed, so they are
warped on the fly and streamed in row-strips, reduced to integer counters
rather than being held in memory.

The grid machinery is shared with figures.py and lives in equal_area.py, so the
console statistics here and the per-ecoregion shares in figures.fig10 are
measured on the same cells. As a cross-check, fig10 reports exactly this
module's `valid_2020` (131,424,956 cells) as its land mask.
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

CHANGE_THRESHOLD = 0.005   # |dHM| at or below this counts as stable
LOW_CUT = 0.10             # "low HM" ceiling / first exceedance level
HIGH_CUT = 0.40            # "moderate HM" ceiling / second exceedance level

FORECAST_YEARS = (2025, 2030, 2035, 2040)
SCENARIOS = ('central', 'upper', 'lower')

ROWS_PER_STRIP = 512       # ~82 MB per raster per strip


def forecast_key(scenario: str, year: int) -> str:
    """config.PATHS key for a blended forecast raster.

    2040 predates the pred_* naming and is stored under hm_*_2040; those files
    are byte-identical to the prediction_2040_*_blended.tif they duplicate.
    """
    return f'hm_{scenario}_2040' if year == 2040 else f'pred_{year}_{scenario}'


# --- equal-area reprojection ------------------------------------------------
# The grid machinery is shared with figures.py. It lives in equal_area, a leaf
# module, so this console script does not import matplotlib/cartopy behind it.

target_grid = equal_area.equal_area_grid
open_layer = equal_area.open_equal_area
AUTHALIC_RADIUS_M = equal_area.AUTHALIC_RADIUS_M


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


def collect() -> dict:
    """Stream every raster once, accumulating the counts each statistic needs."""
    keys = {
        'obs2000': 'hm_2000_aa',
        'obs2020': 'hm_2020_aa',
        **{(s, y): forecast_key(s, y) for s in SCENARIOS for y in FORECAST_YEARS},
    }
    missing = [str(config.PATHS[k]) for k in keys.values()
               if not config.PATHS[k].exists()]
    if missing:
        raise SystemExit('Missing input rasters:\n  ' + '\n  '.join(missing))

    change = {c: dict(increase=0, decrease=0, stable=0, n=0)
              for c in ('projected', 'observed')}
    low = {sc_yr: dict(base=0, crossed=0) for sc_yr in keys if sc_yr not in
           ('obs2000', 'obs2020')}
    mid = {sc_yr: dict(base=0, crossed=0) for sc_yr in low}
    valid_2020 = 0
    # Unrestricted bases (observed 2020 only). Each forecast's own base is a
    # touch smaller wherever that forecast has no value; reported separately.
    low_base_all = mid_base_all = 0

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

            obs2000 = read_strip(src['obs2000'], window)
            obs2020 = read_strip(src['obs2020'], window)
            fin2000 = np.isfinite(obs2000)
            fin2020 = np.isfinite(obs2020)
            valid_2020 += int(np.count_nonzero(fin2020))

            # Bases for the exceedance blocks, evaluated on observed 2020.
            with np.errstate(invalid='ignore'):
                low_base = fin2020 & (obs2020 < LOW_CUT)
                mid_base = fin2020 & (obs2020 >= LOW_CUT) & (obs2020 < HIGH_CUT)
            low_base_all += int(np.count_nonzero(low_base))
            mid_base_all += int(np.count_nonzero(mid_base))

            # Observed change 2000-2020.
            both = fin2000 & fin2020
            with np.errstate(invalid='ignore'):
                delta = obs2020 - obs2000
                change['observed']['increase'] += int(np.count_nonzero(
                    both & (delta > CHANGE_THRESHOLD)))
                change['observed']['decrease'] += int(np.count_nonzero(
                    both & (delta < -CHANGE_THRESHOLD)))
            change['observed']['n'] += int(np.count_nonzero(both))

            for sc_yr in low:
                pred = read_strip(src[sc_yr], window)
                fin = np.isfinite(pred)

                with np.errstate(invalid='ignore'):
                    # Denominator excludes pixels this forecast cannot resolve,
                    # so each percentage is exact even if footprints differ.
                    lb = low_base & fin
                    mb = mid_base & fin
                    low[sc_yr]['base'] += int(np.count_nonzero(lb))
                    low[sc_yr]['crossed'] += int(np.count_nonzero(
                        lb & (pred >= LOW_CUT)))
                    mid[sc_yr]['base'] += int(np.count_nonzero(mb))
                    mid[sc_yr]['crossed'] += int(np.count_nonzero(
                        mb & (pred >= HIGH_CUT)))

                    # Projected change 2020-2040 uses the central 2040 forecast.
                    if sc_yr == ('central', 2040):
                        ok = fin2020 & fin
                        delta = pred - obs2020
                        change['projected']['increase'] += int(np.count_nonzero(
                            ok & (delta > CHANGE_THRESHOLD)))
                        change['projected']['decrease'] += int(np.count_nonzero(
                            ok & (delta < -CHANGE_THRESHOLD)))
                        change['projected']['n'] += int(np.count_nonzero(ok))

        print('\r' + ' ' * 70 + '\r', end='', file=sys.stderr, flush=True)

    for row in change.values():
        row['stable'] = row['n'] - row['increase'] - row['decrease']

    native_km2 = source_land_area_km2()
    warped_km2 = (valid_2020 * config.EQUAL_AREA_RES ** 2 / 1e6
                  if config.EQUAL_AREA_CRS else None)

    return {'grid': (height, width), 'valid_2020': valid_2020,
            'change': change, 'low': low, 'mid': mid,
            'low_base_all': low_base_all, 'mid_base_all': mid_base_all,
            'crs': config.EQUAL_AREA_CRS, 'res': config.EQUAL_AREA_RES,
            'native_km2': native_km2, 'warped_km2': warped_km2}


# --- console report ---------------------------------------------------------

WIDTH = 78
AREA_TOLERANCE_PCT = 2.0    # warped vs native land area disagreement to tolerate


def area_ratio(res: dict) -> float:
    """Percentage by which the warped land area exceeds the native measure."""
    if not res.get('warped_km2') or not res.get('native_km2'):
        return 0.0
    return 100.0 * (res['warped_km2'] / res['native_km2'] - 1.0)


def pct(part: int, whole: int) -> str:
    return '     -  ' if not whole else f'{100.0 * part / whole:7.3f}%'


def rule(char: str = '-') -> str:
    return char * WIDTH


def report(res: dict) -> None:
    height, width = res['grid']
    cells = height * width

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
        print(' equal-area cells, so a pixel share is also an area share')
        if abs(area_ratio(res)) > AREA_TOLERANCE_PCT:
            print()
            print(f' !! WARNING: the two land areas disagree by more than '
                  f'{AREA_TOLERANCE_PCT}%.')
            print(f' !! {res["crs"]} is likely filling destination cells that lie')
            print(f' !! outside its valid domain, which corrupts every percentage')
            print(f' !! below. Prefer a cylindrical equal-area CRS (EPSG:6933).')
    else:
        print(f' grid           EPSG:4326 (native, NOT equal area)')
        print(f'                {height:,} x {width:,}  ({cells:,} cells)')
        print(f' valid in 2020  {res["valid_2020"]:,} pixels '
              f'({100.0 * res["valid_2020"] / cells:.2f}% of grid)')
        print(' percentages are shares of pixels, not of ground area')
    print()

    # --- change direction ---
    print(f'OBSERVED VS PROJECTED CHANGE   (stable = |dHM| <= {CHANGE_THRESHOLD})')
    print(rule())
    print(f'  {"":<34}{"increase":>10}{"decrease":>10}{"stable":>10}{"n pixels":>14}')
    for label, key in ((f'Projected 2020-2040 (central)', 'projected'),
                       (f'Observed  2000-2020', 'observed')):
        row = res['change'][key]
        print(f'  {label:<34}{pct(row["increase"], row["n"]):>10}'
              f'{pct(row["decrease"], row["n"]):>10}'
              f'{pct(row["stable"], row["n"]):>10}{row["n"]:>14,}')
    print()

    # --- exceedance blocks ---
    for title, block, base_text, level, base_n in (
        (f'HM EXCEEDING {LOW_CUT:.2f}', res['low'],
         f'pixels with observed HM < {LOW_CUT:.2f} in 2020', LOW_CUT,
         res['low_base_all']),
        (f'HM EXCEEDING {HIGH_CUT:.2f}', res['mid'],
         f'pixels with {LOW_CUT:.2f} <= observed HM < {HIGH_CUT:.2f} in 2020',
         HIGH_CUT, res['mid_base_all']),
    ):
        print(f'{title}')
        print(f'  base: {base_text}   (n = {base_n:,})')
        # Each cell divides by its own forecast's base, which drops pixels that
        # forecast leaves unresolved. Flag it when that shrinks the denominator.
        gap = max(base_n - v['base'] for v in block.values())
        if gap:
            print(f'  denominators are per-forecast; the largest excludes '
                  f'{gap:,} pixels ({100.0 * gap / base_n:.4f}%) with no forecast value')
        print(rule())
        header = ''.join(f'{y:>10}' for y in FORECAST_YEARS)
        print(f'  {"% of base reaching HM >= " + f"{level:.2f}":<34}{header}')
        for scenario in SCENARIOS:
            cells_out = ''.join(
                pct(block[(scenario, y)]['crossed'], block[(scenario, y)]['base'])
                .rjust(10) for y in FORECAST_YEARS)
            print(f'  {scenario:<34}{cells_out}')
        print()

    print(rule('='))


def main() -> None:
    where = (f'warping to {config.EQUAL_AREA_CRS} @ {config.EQUAL_AREA_RES} m'
             if config.EQUAL_AREA_CRS else 'on the native EPSG:4326 grid')
    print(f'Streaming 14 rasters, {where}...', file=sys.stderr)
    report(collect())


if __name__ == '__main__':
    main()
