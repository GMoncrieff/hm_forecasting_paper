"""Single entry-point: regenerate every enabled figure and table.

Edit config.RUN to choose outputs, then run:

    mamba run -n hm_plots python make_paper.py

Outputs land in config.OUTPUT_DIR (default: ./output). Figure 10 is ordered
before Tables S2-S9 because the tables consume its unprotected_loss_stats.csv.
"""
import time
import traceback

import config
import figures
import tables

# (name, callable) in dependency order; fig10 must precede tables_s2_s9.
ORDERED_OUTPUTS = [
    ("fig2_3", figures.fig2_3),
    ("figS1_S4", figures.figS1_S4),
    ("fig4_5", figures.fig4_5),
    ("figSX", figures.figSX),
    ("fig6", figures.fig6),
    ("fig7", figures.fig7),
    ("fig8_9", figures.fig8_9),
    ("figS5", figures.figS5),
    ("figS6", figures.figS6),
    ("fig10", figures.fig10),
    ("table_s1", tables.table_s1),
    ("tables_s2_s9", tables.tables_s2_s9),
]


def run():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = None
    if config.USE_DASK:
        try:
            import utils
            client = utils.make_dask_client()
            print(f"[dask] {client}")
        except Exception as exc:  # noqa: BLE001
            print(f"[dask] not started ({exc}); using default scheduler")

    ran, skipped, failed = [], [], {}
    for name, fn in ORDERED_OUTPUTS:
        if not config.RUN.get(name, False):
            skipped.append(name)
            print(f"[skip] {name}")
            continue
        print(f"[run ] {name} ...", flush=True)
        t0 = time.time()
        try:
            fn()
            ran.append(name)
            print(f"[done] {name}  ({time.time() - t0:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            failed[name] = repr(exc)
            print(f"[FAIL] {name}: {exc}")
            traceback.print_exc()
            if config.STOP_ON_ERROR:
                break

    if client is not None:
        client.close()

    print("\n=== summary ===")
    print(f"ran:     {ran}")
    print(f"skipped: {skipped}")
    print(f"failed:  {list(failed)}")
    return {"ran": ran, "skipped": skipped, "failed": failed}


if __name__ == "__main__":
    run()
