"""Central configuration: paths, parameters, colours, and run toggles.

Edit RUN[...] = False to skip an output. Edit DATA_DIR / PATHS for relocation.
"""
from pathlib import Path

# --- locations ---
REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR.parent / "data"     # reuse existing ../data (no duplication)
V2_DIR = REPO_DIR.parent / "v2"         # legacy dir holding the covariates xlsx
OUTPUT_DIR = REPO_DIR / "output"

# --- which outputs to generate (edit to disable) ---
RUN = {
    "fig2_3": True, "figS1_S4": True, "fig4_5": True, "figSX": True,
    "fig6": True, "fig7": True, "fig8_9": True, "figS5": True,
    "figS6": True, "fig10": True, "table_s1": True, "tables_s2_s9": True,
}

STOP_ON_ERROR = False
USE_DASK = True

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
    "hm_2000_aa": DATA_DIR / "HM_2000_AA_1000.tiff",
    "hm_2005_aa": DATA_DIR / "HM_2005_AA_1000.tiff",
    "hm_2010_aa": DATA_DIR / "HM_2010_AA_1000.tiff",
    "hm_2015_aa": DATA_DIR / "HM_2015_AA_1000.tiff",
    "hm_2020_aa": DATA_DIR / "HM_2020_AA_1000.tiff",
    "hm_1990_aa": DATA_DIR / "HM_1990_AA_1000.tiff",
    "hm_1995_aa": DATA_DIR / "HM_1995_AA_1000.tiff",
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
    "pred_2005_lower": DATA_DIR / "prediction_2005_lower_blended.tif",
    "pred_2010_lower": DATA_DIR / "prediction_2010_lower_blended.tif",
    "pred_2015_lower": DATA_DIR / "prediction_2015_lower_blended.tif",
    "pred_2020_lower": DATA_DIR / "prediction_2020_lower_blended.tif",
    "covariates_xlsx": V2_DIR / "Supplementary_Table_S2_covariates.xlsx",
}
# NOTE: fig10 / S5 / S6 reference further inputs (PA raster, ecoregions, 2040
# blended). Add any further paths the ported bodies require, here, during
# Tasks 10-11.

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
INSET_SIZE = 0.085

# --- colours (verbatim from plot_fig2_fig3.py) ---
COOLWARM_STOPS = [
    (0, '#3952c4'), (0.125, '#6c8ef0'), (0.25, '#b2ccfa'), (0.375, '#CFCEDE'),
    (0.5, '#E4E4E4'), (0.6, '#D9A298'), (0.7, '#D07D75'), (0.8, '#c73534'),
    (0.9, '#9D0E1F'), (1.00, '#7E0414'),
]
CLASS_COLORS = ['#000000', '#bee6c2', '#ffbb00', '#ff0000', '#dd87ff']
CLASS_LEGEND_LABELS = [
    'Non-natural 2020',
    'Still Natural 2040',
    'Natural lands loss 2040 (upper 97.5% forecast)',
    'Natural lands loss 2040 (central 50% forecasts)',
    'Natural lands loss 2040 (lower 2.5% forecast)',
]

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
FIGS_YEAR_VARIANTS = [
    ("pred_2025_central", "HM change 2025-2020", "figS2_hm2025_map.png"),
    ("pred_2030_central", "HM change 2030-2020", "figS3_hm2030_map.png"),
    ("pred_2035_central", "HM change 2035-2020", "figS4_hm2035_map.png"),
]

# --- regional zooms (Figs S5/S6) ---
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
TARGET_PX_ZOOM = 1200
