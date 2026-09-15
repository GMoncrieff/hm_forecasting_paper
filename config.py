"""Central configuration: paths, parameters, colours, and run toggles.

Edit RUN[...] = False to skip an output. Edit DATA_DIR / PATHS for relocation.
"""
from pathlib import Path

# --- locations ---
REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR / "data"     # reuse existing ../data (no duplication)
CONV_DIST_DIR = DATA_DIR / "conv_dist"   # distributional-model outputs
# `output` is a symlink into iCloud Drive, which the sandbox denies and
# which evicts large files to dataless placeholders mid-run; outputs go to
# a real local directory instead.
OUTPUT_DIR = REPO_DIR / "output_new"

# --- which outputs to generate (edit to disable) ---
RUN = {
    "fig2_3": True, "figS1_S4": True, "fig4_5": True, "figSX": True,
    "fig6": True, "fig7": True, "fig8_9": True, "figS5": True,
    "figS6": True, "fig10": True, "table_1": True, "tables_s1_s8": True,
}

STOP_ON_ERROR = False
USE_DASK = False

# --- table output format ---
# one of: "pdf", "docx", "both"  (applies to table_1 and tables_s1_s8)
TABLE_FORMAT = "pdf"

# --- equal-area analysis grid ---
# Source rasters are EPSG:4326, where cell ground area falls off with cos(lat),
# so a raw pixel count over-weights high latitudes. Every quantity the paper
# reports is therefore computed after warping onto this shared equal-area grid,
# which makes a pixel share an area share: stats.py (all console statistics),
# figures.fig10 (-> unprotected_loss_stats.csv -> Tables S1-S8), figures.fig4_5
# (Figs 4/5 densities) and figures.fig8_9 (the sample behind the Spearman
# matrix, Fig 8's panels and Fig 9's colour cut-points). Maps are NOT warped -
# cartopy renders them area-honestly from the native grid.
#
# Any equal-area CRS gives the same percentages; this one is the NSIDC EASE-Grid
# 2.0 Global standard. Prefer a *cylindrical* equal-area CRS: pseudocylindrical
# ones (e.g. Equal Earth, EPSG:8857) keep out-of-domain corners in their
# bounding rectangle, and stats.py will warn if the land area disagrees.
# Set EQUAL_AREA_CRS = None to count on the native EPSG:4326 grid instead.
EQUAL_AREA_CRS = "EPSG:6933"
EQUAL_AREA_RES = 1000            # metres, in EQUAL_AREA_CRS units
# "nearest" keeps the HM value distribution intact, so only the area weighting
# changes. Averaging/bilinear would smooth values across the 0.10/0.40 cuts.
#
# This applies to the probability rasters too, even though "average" would be
# the integral-preserving choice for a field that is about to be summed. Every
# probability is summed over a base mask read off the *observed* HM raster, and
# only nearest guarantees a destination cell draws from the same source cell in
# both - so base and probability stay on identical ground. Mixing the two
# methods would decouple them, which is a worse error than the resampling one.
EQUAL_AREA_RESAMPLING = "nearest"

# --- condition-class cut-points ---------------------------------------------
# Shared by stats.py, figures.fig2_3/fig6/fig10 and the probability rasters
# themselves (HM_p10_* is P(HM >= LOW_CUT), HM_p40_* is P(HM >= HIGH_CUT)).
# Hoisted here because three modules used to carry their own copy.
#
# The cuts are CLOSED at the bottom: land is "natural" when HM <= LOW_CUT, not
# HM < LOW_CUT, and "moderately modified" over (LOW_CUT, HIGH_CUT). Every base
# mask in the repo uses that convention, so the three 2020 classes still
# partition the land exactly. Note this makes the natural base and the
# exceedance event it is scored against overlap at exactly HM == LOW_CUT: a
# pixel sitting on 0.10 in 2020 counts as natural and has probability 1 of
# being at or above 0.10, which is the intended reading of a closed cut.
LOW_CUT = 0.10             # "natural" ceiling, inclusive
HIGH_CUT = 0.40            # "moderately modified" ceiling

FORECAST_YEARS = (2025, 2030, 2035, 2040)   # forecast store, base year 2020
HINDCAST_YEARS = (2005, 2010, 2015, 2020)   # hindcast store, base year 2000

# Risk brackets applied to P(HM >= LOW_CUT) in 2040 (Fig 10, Tables S1-S8).
P_LOSS_BREAKS = (0.025, 0.5)
PROTECTION_TARGET = 30.0   # percent of an ecoregion, the 30x30 target

# --- input paths (logical name -> file) ---
PATHS = {
    "hm_diff": DATA_DIR / "HM_DIFF.tif",
    "hm_diff_obs": DATA_DIR / "hm_diff_obs.tif",
    "raster_classes": DATA_DIR / "raster_classes.tif",
    "hm_upper_2040": DATA_DIR / "HM_upper_2040.tif",
    "hm_lower_2040": DATA_DIR / "HM_lower_2040.tif",
    "hm_central_2040": DATA_DIR / "HM_central_2040.tif",
    "hm_observed_2020": DATA_DIR / "HM_observed_2020.tiff",
    "esri_hm": DATA_DIR / "ESRI_HM.tif",
    "cpi_hm": DATA_DIR / "CPI_HM.tif",
    "split_mask": DATA_DIR / "split_mask_1000.tif",
    "hm_static_iucn_strict": DATA_DIR / "hm_static_iucn_strict_1000.tiff",
    "hm_2000_aa": DATA_DIR / "HM_observed_2000.tiff",
    "hm_2005_aa": DATA_DIR / "HM_observed_2005.tiff",
    "hm_2010_aa": DATA_DIR / "HM_observed_2010.tiff",
    "hm_2015_aa": DATA_DIR / "HM_observed_2015.tiff",
    "hm_2020_aa": DATA_DIR / "HM_observed_2020.tiff",
    "hm_1990_aa": DATA_DIR / "HM_observed_1990.tiff",
    "hm_1995_aa": DATA_DIR / "HM_observed_1995.tiff",
    "pred_2005_central": DATA_DIR / "prediction_2005_central_blended.tif",
    "pred_2010_central": DATA_DIR / "prediction_2010_central_blended.tif",
    "pred_2015_central": DATA_DIR / "prediction_2015_central_blended.tif",
    "pred_2020_central": DATA_DIR / "prediction_2020_central_blended.tif",
    "pred_2025_central": DATA_DIR / "prediction_2025_central_blended.tif",
    "pred_2030_central": DATA_DIR / "prediction_2030_central_blended.tif",
    "pred_2035_central": DATA_DIR / "prediction_2035_central_blended.tif",
    "pred_2005_upper": DATA_DIR / "prediction_2005_upper_blended.tif",
    "pred_2010_upper": DATA_DIR / "prediction_2010_upper_blended.tif",
    "pred_2015_upper": DATA_DIR / "prediction_2015_upper_blended.tif",
    "pred_2020_upper": DATA_DIR / "prediction_2020_upper_blended.tif",
    "pred_2025_upper": DATA_DIR / "prediction_2025_upper_blended.tif",
    "pred_2030_upper": DATA_DIR / "prediction_2030_upper_blended.tif",
    "pred_2035_upper": DATA_DIR / "prediction_2035_upper_blended.tif",
    "pred_2005_lower": DATA_DIR / "prediction_2005_lower_blended.tif",
    "pred_2010_lower": DATA_DIR / "prediction_2010_lower_blended.tif",
    "pred_2015_lower": DATA_DIR / "prediction_2015_lower_blended.tif",
    "pred_2020_lower": DATA_DIR / "prediction_2020_lower_blended.tif",
    "pred_2025_lower": DATA_DIR / "prediction_2025_lower_blended.tif",
    "pred_2030_lower": DATA_DIR / "prediction_2030_lower_blended.tif",
    "pred_2035_lower": DATA_DIR / "prediction_2035_lower_blended.tif",
    "covariates_xlsx": DATA_DIR / "Supplementary_Table_S2_covariates.xlsx",
    # Per-pixel exceedance probabilities derived from the 64-level quantile
    # function: HM_p10_YYYY is P(HM >= LOW_CUT), HM_p40_YYYY is P(HM >= HIGH_CUT).
    # 2005-2020 come from the hindcast (base year 2000), 2025-2040 from the
    # forecast (base year 2020). Same grid and nodata as the observed rasters.
    **{f"hm_p10_{y}": CONV_DIST_DIR / "cogs" / f"HM_p10_{y}.tif"
       for y in (2005, 2010, 2015, 2020, 2025, 2030, 2035, 2040)},
    **{f"hm_p40_{y}": CONV_DIST_DIR / "cogs" / f"HM_p40_{y}.tif"
       for y in (2005, 2010, 2015, 2020, 2025, 2030, 2035, 2040)},
    # The full quantile functions. int16 with `scale`/`sentinel` attributes -
    # NOT the CF names xarray decodes, so read them through quantiles.decode().
    "qf_forecast": CONV_DIST_DIR / "forecast" / "forecast_qf.icechunk",
    "qf_hindcast": CONV_DIST_DIR / "hindcast" / "hindcast_qf.icechunk",
}
# NOTE: fig10 / S5 / S6 reference further inputs (PA raster, ecoregions, 2040
# blended). Add any further paths the ported bodies require, here, during
# Tasks 10-11.

# --- Table 1: covariate rows not in the source workbook ---------------------
# The distributional model adds long-range context channels, precomputed on the
# full raster (computing them inside a <=128 px training chip saturates them).
# They are appended here rather than edited into the workbook because data/ is
# regenerated wholesale from the model repo, so an xlsx edit would be lost on
# the next refresh. Keys match the workbook's own column names.
_HM_TS_CITATION = 'Theobald et al. (2025)'
_HM_TS_LINK = 'https://zenodo.org/records/14449495'
EXTRA_COVARIATES = [
    {
        'Covariate': 'Distance to past HM change',
        'Type': 'Dynamic',
        'Source dataset': 'Derived from the global cumulative HM time '
                          'series: Euclidean distance to the nearest pixel that '
                          'changed over the preceding interval',
        'Native spatial resolution': '1 km',
        'Native temporal resolution': '5-year intervals',
        'Time period used': '1990-2020',
        'Units': 'Pixels (1 km)',
        'Citation': _HM_TS_CITATION,
        'Link / DOI': _HM_TS_LINK,
    },
    {
        'Covariate': 'Past HM-change density at radii 3, 30, 100 px',
        'Type': 'Dynamic',
        'Source dataset': 'Derived from the global cumulative HM time '
                          'series: fraction of pixels within each radius that '
                          'changed over the preceding interval',
        'Native spatial resolution': '1 km',
        'Native temporal resolution': '5-year intervals',
        'Time period used': '1990-2020',
        'Units': 'Unitless (0-1)',
        'Citation': _HM_TS_CITATION,
        'Link / DOI': _HM_TS_LINK,
    },
    {
        'Covariate': 'Neighbourhood mean HM at radii 3, 30, 100 px',
        'Type': 'Dynamic',
        'Source dataset': 'Derived from the global cumulative HM time '
                          'series: mean HM within each radius',
        'Native spatial resolution': '1 km',
        'Native temporal resolution': '5-year intervals',
        'Time period used': '1990-2020',
        'Units': 'Unitless (0-1)',
        'Citation': _HM_TS_CITATION,
        'Link / DOI': _HM_TS_LINK,
    },
    {
        'Covariate': 'Neighbourhood maximum HM at radii 3, 30, 100 px',
        'Type': 'Dynamic',
        'Source dataset': 'Derived from the global cumulative HM time '
                          'series: maximum HM within each radius',
        'Native spatial resolution': '1 km',
        'Native temporal resolution': '5-year intervals',
        'Time period used': '1990-2020',
        'Units': 'Unitless (0-1)',
        'Citation': _HM_TS_CITATION,
        'Link / DOI': _HM_TS_LINK,
    },
]

# --- DPI ---
DPI_MAP = 600
DPI_PLOT = 300

# --- global Robinson map params (Figs 2, 3, S1-S4, 8/9) ---
CLIM = (-0.2, 0.2)
OCEAN_HEX = "#F4FCFF"
INSET_DEFS = [
    {"center": (-3.046461, -49.938504), "anchor": (-30, -105)},   # Para, Brazil
    {"center": (0.232389, 37.375075), "anchor": (-30, -13)},      # Northern Kenya
    {"center": (9.921023, 77.617712), "anchor": (-30, 77)},       # Southern India
]
RADIUS_DEG = 2.0
# Inset diameter as a fraction of the MAIN MAP AXES' width. This was a figure
# fraction (0.085), which only gave the intended circle on the 4x4 in canvas
# hv.render() returns for Fig 2; Figs 3a/3b build their own 13.9 x 7.6 in canvas
# and got insets both oversized and elliptical. 0.085 of Fig 2's figure is
# 0.34 in, and its map axes are 0.70 * 4 = 2.8 in wide, so this reproduces
# Fig 2 exactly while making every other map match it. See
# utils.add_circular_insets.
INSET_MAP_FRAC = 0.085 / 0.70

# Fig 2's map axes width in inches (0.70 of its 4 in canvas), the reference every
# other map's strokes are scaled against. Line widths and marker sizes are in
# POINTS - an absolute unit - so identical values on a wider figure draw thinner
# once both are reproduced at one page width. Fig 3's map is 8.94 in, so its
# locator rings and inset borders came out 3.2x too fine. The four sizes below
# are Fig 2's, and utils.add_circular_insets multiplies them by
# map_width / INSET_REF_MAP_W_IN, which is exactly 1 for Fig 2 and Figs S1-S6.
INSET_REF_MAP_W_IN = 0.70 * 4.0
INSET_BORDER_LW = 0.3      # ring around each inset
INSET_SPINE_LW = 0.3       # the map's own frame
INSET_MARKER_SIZE = 4.0    # locator ring on the main map
INSET_MARKER_LW = 0.4      # its stroke

# --- colours (verbatim from plot_fig2_fig3.py) ---
COOLWARM_STOPS = [
    (0, '#3952c4'), (0.125, '#6c8ef0'), (0.25, '#b2ccfa'), (0.375, '#CFCEDE'),
    (0.5, '#E4E4E4'), (0.6, '#D9A298'), (0.7, '#D07D75'), (0.8, '#c73534'),
    (0.9, '#9D0E1F'), (1.00, '#7E0414'),
]
# Retained for the retired 2040 categorical map (data/raster_classes.tif). No
# current figure reads either; Figure 3 and Figure S6 now show exceedance
# probabilities instead. utils.register_class_cmap still registers the palette.
CLASS_COLORS = ['#000000', '#bee6c2', '#ffbb00', '#ff0000', '#dd87ff']

# --- exceedance-probability scale (Figs 3, 6, S6) ---------------------------
# Sampled from output/p_exceed_idea.png. The breaks are deliberately uneven:
# a linear scale puts almost every land pixel in the first colour, because most
# of the world has a small probability of crossing a threshold and the signal
# lives in the 0.01-0.2 range. Drives a BoundaryNorm, not a Normalize.
P_EXCEED_LEVELS = [0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0]
P_EXCEED_COLORS = [
    '#e8efea',   # 0     - 0.01
    '#c6d8ce',   # 0.01  - 0.025
    '#f6e3b0',   # 0.025 - 0.05
    '#efc978',   # 0.05  - 0.1
    '#e5a24f',   # 0.1   - 0.2
    '#d4703a',   # 0.2   - 0.35
    '#b8432a',   # 0.35  - 0.5
    '#7e1f12',   # 0.5   - 1
]
# Land outside a map's base - already above the threshold in the base year, so
# the probability is not defined for it - has to read as neither ocean nor a
# low probability. This warm beige is the one output/p_exceed_idea.png uses;
# it separates from the cool pale sage of the lowest band by hue rather than by
# lightness, which is what makes the two legible against each other at map
# scale. A neutral grey here is nearly indistinguishable from that band.
OUT_OF_BASE_HEX = '#e4e1d8'

# --- Figure 10 palettes ------------------------------------------------------
# Sampled from output/radial_fig_idea.png. Radial stack order is centre -> rim.
RADIAL_COLORS = {
    'protected':    '#2c7c6f',
    'p_low':        '#cbd3cb',   # unprotected natural, P(loss) < 0.025
    'p_mid':        '#efb05e',   # unprotected natural, 0.025 <= P(loss) < 0.5
    'p_high':       '#b8432a',   # unprotected natural, P(loss) >= 0.5
    'non_natural':  '#8a7f76',
}
RADIAL_LABELS = {
    'protected':   'Protected in 2020',
    'p_low':       'Natural, unprotected - P(loss) < 0.025',
    'p_mid':       'Natural, unprotected - P(loss) 0.025-0.5',
    'p_high':      'Natural, unprotected - P(loss) > 0.5',
    'non_natural': 'Non-natural in 2020',
}

# Risk of missing the 30% protection target, in the order fig10 tests them.
TARGET_CLASSES = ('already met', 'feasible', 'tight', 'at risk',
                  'infeasible on natural land')
TARGET_CLASS_COLORS = {
    'already met':                '#2c7c6f',
    'feasible':                   '#7fa9a0',
    'tight':                      '#efb05e',
    'at risk':                    '#b8432a',
    'infeasible on natural land': '#8a7f76',
}
TARGET_CLASS_DESCRIPTIONS = {
    'already met':                'protected >= 30% in 2020',
    'feasible':                   'target met on land with P(loss) < 0.025',
    'tight':                      'target needs land with P(loss) up to 0.5',
    'at risk':                    'target needs land with P(loss) > 0.5',
    'infeasible on natural land': 'too little natural land remains in 2020',
}
# Classes whose legend entry keeps its class name as a prefix. For the other
# three the description already reads as a full label ("Target met on land
# with P(loss) < 0.025"), and the prefix only repeated it in shorthand; these
# two do not stand alone without it.
TARGET_CLASS_LABEL_PREFIX = ('already met', 'infeasible on natural land')

# --- Figure 7 quantile fan ---------------------------------------------------
# Opacity encodes how central a value is: w = 1 - 2|F - 0.5|, so the median is
# opaque and the tails fade out. The legend in utils.plot_quantile_fan_legend
# reproduces this exact mapping, so the two must be changed together.
FAN_COLOR = 'steelblue'
FAN_GAMMA = 0.6
FAN_ALPHA_MIN = 0.05
FAN_ALPHA_MAX = 0.95
FAN_NY = 600
FAN_NT = 600

# Figure 7's standalone fan legend. The bar encodes nothing across its
# thickness - only along it - so it is drawn thin, and the space goes to the
# labels, which are the part a reader actually reads. BAR_FRAC is the bar's
# thickness as a fraction of the figure's short side.
FAN_LEGEND_FONTSIZE = 15
FAN_LEGEND_BAR_FRAC = 0.16

# x-limits of Figure 7's bottom row, the predictive densities for 2020. One
# fixed window across all four panels so their widths are comparable; the lower
# edge sits below zero so a density piled against the HM floor is not clipped
# by the spine.
FIG7_DENSITY_XLIM = (-0.03, 0.5)

# --- validation sample for Figs 4 and 5 --------------------------------------
# The previous model used a spatial split: some tiles trained, others were held
# out, and `split_mask == 2` was the held-out set. The distributional model uses
# k-fold cross-validation and stitches holdout predictions, so EVERY pixel is
# predicted by a model that never saw it and the whole grid is out of sample.
#
# Restricting to one split value would now discard ~90% of a fully valid sample
# without buying any independence. Set this back to 2 to reproduce the old
# figures on the old sample.
VALIDATION_SPLIT = None    # or 2, to use only split_mask == 2

# --- Figure 5 predictive-mixture sample --------------------------------------
# Chunk-aligned 512x512 tiles drawn from the hindcast quantile store; each is
# exactly one icechunk chunk, so this is one read per tile rather than four.
#
# The count matters more than it looks. HM and its forecast errors are strongly
# spatially correlated, so the ~260k pixels inside one tile behave as close to a
# single draw: one tile's PIT histogram spikes above 7 in a single bin. The
# effective sample size of Figure 5 is therefore closer to the number of TILES
# than to the number of pixels, which is why the independence error bars in
# panel (a) are a floor rather than an estimate.
#
# Raising this from 48 to 200 did NOT smooth panel (b) - the histogram keeps its
# shape - which is how we know that shape is real calibration structure and not
# sampling noise. 200 tiles is ~32 M pixels and about 76 s.
FIG5_N_BLOCKS = 200
FIG5_SEED = 11

# Bins in Figure 5's PIT histogram. Coarse on purpose: the effective sample
# size is closer to the tile count than to the pixel count (see above), so a
# fine histogram resolves structure the sample cannot support. 7 keeps the
# expected count per bin comfortably above the tile count.
FIG5_PIT_BINS = 7

# --- ternary scheme (Figs 8/9) — verbatim from plot_fig8_fig9.py ---
OCEAN_RGB = (244 / 255, 252 / 255, 255 / 255)
BACKGROUND_RGB = (0.941, 0.941, 0.941)
TERNARY_N_BINS = 4
TERNARY_COLORS = [
    [0.851, 0.647, 0.129],  # ESRI -> gold
    [0.122, 0.824, 0.796],  # HM -> cyan
    [0.949, 0.247, 0.867],  # CPI -> magenta
]
TERNARY_LABELS = ('ESRI', 'HM', 'CPI')
TRANSPARENT_Q = 0.75
ALPHA_LEVELS = (0.40, 0.60, 0.80, 1.00)
BASE_ALPHA = 0.0

# --- multi-image figure counts (were range(100) / 10 in the notebooks) ---
FIG7_SEEDS = [0]        # one timeseries figure per seed
FIGSX_N_PLOTS = 1       # number of random 2-site panels

# --- colorbar labels for the year-variant maps (Figs S2/S3/S4) ---
# (pred_path_key, colorbar_label, output_filename)
# "Expected" is literal: the blended central surface is E[Q], the mean of the
# predictive distribution, so these differences are expectations rather than a
# median scenario. Fig S1 keeps the plain wording because it shows observations.
FIGS_YEAR_VARIANTS = [
    ("pred_2025_central", "Expected HM change 2025-2020", "figS2_hm2025_map.png"),
    ("pred_2030_central", "Expected HM change 2030-2020", "figS3_hm2030_map.png"),
    ("pred_2035_central", "Expected HM change 2035-2020", "figS4_hm2035_map.png"),
]

# --- Figures S5/S6: the forecast's lower and upper bounds -------------------
# Both are now single global maps in the Figure 2 mould - same coolwarm ramp,
# same CLIM, same three circular insets - so the three panels read as one
# triptych: S5 is the 2.5th percentile of each pixel's predictive distribution,
# Figure 2 its mean, S6 the 97.5th. They were 11 regional zooms each.
# (bound, colorbar_label, output_filename)
FIGS_BOUND_VARIANTS = [
    ("lower", "Lower HM change 2020-2040", "figS5_hmdiff_lower_map.png"),
    ("upper", "Upper HM change 2020-2040", "figS6_hmdiff_upper_map.png"),
]

# --- regional zooms ---
# No figure reads these now that S5/S6 are global maps; kept because the bboxes
# are the only record of the regional framing and are cheap to hold.
REGIONS = {
    "north_america":     ("North America",                (-168, -52, 14, 72)),
    "central_america":   ("Central America",              (-93, -59, 7, 22)),
    "south_america":     ("South America",                (-82, -34, -56, 13)),
    "europe":            ("Europe",                       (-25, 45, 34, 72)),
    "mena":              ("Middle East & North Africa",   (-18, 63, 11, 40)),
    "central_asia":      ("Central Asia",                 (40, 120, 28, 79)),
    "east_asia":         ("East Asia",                    (100, 146, 18, 54)),
    "south_asia":        ("South Asia",                   (60, 98, 5, 38)),
    "southeast_asia":    ("Southeast Asia",               (92, 142, -11, 29)),
    "subsaharan_africa": ("Sub-Saharan Africa",           (-19, 52, -36, 18)),
    "australia_nz":      ("Australia & New Zealand",      (110, 179, -48, -10)),
}
TARGET_PX_ZOOM = 1200      # decimation target for a regional render; unused
