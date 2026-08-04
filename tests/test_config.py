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
    assert config.OUTPUT_DIR.name == "output"
    assert config.DPI_MAP == 600 and config.DPI_PLOT == 300
    assert config.CLIM == (-0.2, 0.2)
    assert len(config.INSET_DEFS) == 3
    assert len(config.COOLWARM_STOPS) == 10
    assert len(config.CLASS_COLORS) == 5
    assert config.FIGSX_N_PLOTS >= 1 and len(config.FIG7_SEEDS) >= 1
