import config
import make_paper


def test_ordered_outputs_cover_run_keys():
    names = [name for name, _ in make_paper.ORDERED_OUTPUTS]
    assert set(names) == set(config.RUN)
    assert names.index("fig10") < names.index("tables_s1_s8")


def test_run_one_only(monkeypatch):
    monkeypatch.setattr(config, "RUN", {k: (k == "table_1") for k in config.RUN})
    monkeypatch.setattr(config, "USE_DASK", False)
    summary = make_paper.run()
    assert summary["ran"] == ["table_1"]
    assert "table_1" not in summary["failed"]
