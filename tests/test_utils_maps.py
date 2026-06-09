import numpy as np
import xarray as xr

import utils


def test_register_cmaps_idempotent():
    utils.register_coolwarm_cmap(); utils.register_coolwarm_cmap()
    utils.register_class_cmap(); utils.register_class_cmap()
    import matplotlib.pyplot as plt
    assert "my_custom_coolwarm" in plt.colormaps()
    assert "raster_classes" in plt.colormaps()


def test_build_global_robinson_map_returns_fig():
    da = xr.DataArray(
        np.random.default_rng(0).uniform(-0.2, 0.2, size=(20, 40)).astype("float32"),
        coords={"y": np.linspace(80, -80, 20), "x": np.linspace(-179, 179, 40)},
        dims=["y", "x"],
    )
    utils.register_coolwarm_cmap()
    fig, ax = utils.build_global_robinson_map(
        da, cmap="my_custom_coolwarm", clim=(-0.2, 0.2),
        cbar_label="test", hide_geo_spine=False,
    )
    assert fig is not None and ax is not None
    import matplotlib.pyplot as plt
    plt.close(fig)
