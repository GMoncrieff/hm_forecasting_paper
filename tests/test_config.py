import pytest

import config
import tables


def test_table_format_is_valid():
    assert config.TABLE_FORMAT in tables.VALID_TABLE_FORMATS


@pytest.mark.parametrize("value,expected", [
    ("pdf", ("pdf",)),
    ("docx", ("docx",)),
    ("both", ("pdf", "docx")),
    ("  DOCX ", ("docx",)),          # case/whitespace tolerant
])
def test_formats_dispatch(monkeypatch, value, expected):
    monkeypatch.setattr(config, "TABLE_FORMAT", value)
    assert tables._formats() == expected


@pytest.mark.parametrize("value", ["", "word", "pdf,docx", "none"])
def test_formats_rejects_unknown(monkeypatch, value):
    monkeypatch.setattr(config, "TABLE_FORMAT", value)
    with pytest.raises(ValueError, match="TABLE_FORMAT"):
        tables._formats()


def test_run_keys_match_known_outputs():
    expected = {
        "fig2_3", "figS1_S4", "fig4_5", "figSX", "fig6", "fig7",
        "fig8_9", "figS5", "figS6", "fig10", "table_1", "tables_s1_s8",
    }
    assert set(config.RUN) == expected
    assert all(isinstance(v, bool) for v in config.RUN.values())


def test_core_paths_and_params_present():
    assert config.DATA_DIR.name == "data"
    # OUTPUT_DIR is a user setting (it has been pointed at output_new, and at a
    # symlink into iCloud before that), so pin where it lives, not what it is
    # called.
    assert config.OUTPUT_DIR.parent == config.REPO_DIR
    assert config.DPI_MAP == 600 and config.DPI_PLOT == 300
    assert config.CLIM == (-0.2, 0.2)
    assert len(config.INSET_DEFS) == 3
    assert len(config.COOLWARM_STOPS) == 10
    assert len(config.CLASS_COLORS) == 5
    assert config.FIGSX_N_PLOTS >= 1 and len(config.FIG7_SEEDS) >= 1


def test_probability_and_quantile_paths_present():
    """Every year of both exceedance levels, plus the two quantile stores."""
    years = config.HINDCAST_YEARS + config.FORECAST_YEARS
    for y in years:
        for level in ("p10", "p40"):
            assert f"hm_{level}_{y}" in config.PATHS, (level, y)
    assert "qf_forecast" in config.PATHS
    assert "qf_hindcast" in config.PATHS
    assert config.FORECAST_YEARS == (2025, 2030, 2035, 2040)
    assert config.HINDCAST_YEARS == (2005, 2010, 2015, 2020)


def test_exceedance_scale_is_consistent():
    """One colour per interval, or the BoundaryNorm silently reuses colours."""
    assert len(config.P_EXCEED_COLORS) == len(config.P_EXCEED_LEVELS) - 1
    assert config.P_EXCEED_LEVELS == sorted(config.P_EXCEED_LEVELS)
    assert config.P_EXCEED_LEVELS[0] == 0.0
    assert config.P_EXCEED_LEVELS[-1] == 1.0


def test_cut_points_and_risk_breaks_are_ordered():
    assert 0.0 < config.LOW_CUT < config.HIGH_CUT < 1.0
    lo, hi = config.P_LOSS_BREAKS
    assert 0.0 < lo < hi < 1.0
    assert 0.0 < config.PROTECTION_TARGET < 100.0


def test_radial_palette_covers_every_stacked_segment():
    """The five segments partition an ecoregion; each needs its own colour."""
    assert set(config.RADIAL_COLORS) == set(config.RADIAL_LABELS)
    assert len(config.RADIAL_COLORS) == 5


def test_extra_covariates_carry_every_workbook_column():
    """A missing key renders as a blank cell rather than raising."""
    required = {"Covariate", "Type", "Source dataset",
                "Native spatial resolution", "Native temporal resolution",
                "Time period used", "Units", "Citation", "Link / DOI"}
    for row in config.EXTRA_COVARIATES:
        assert required <= set(row), sorted(required - set(row))
        assert row["Type"] in {"Static", "Dynamic", "Static (learned)"}
