import numpy as np

import config
import utils


def test_fmt_coord_hemispheres():
    assert utils.fmt_coord(-10.1163, 32.1712) == ("10.12°S", "32.17°E")
    assert utils.fmt_coord(9.92, -77.62) == ("9.92°N", "77.62°W")


def test_alpha_percentile_points():
    pts = utils._alpha_percentile_points(0.75, (0.4, 0.6, 0.8, 1.0))
    assert np.allclose(pts, [75.0, 81.25, 87.5, 93.75])


def test_ternary_alpha_rgb_shape_and_invalid():
    v = np.array([[0.2, 0.8], [0.5, np.nan]], dtype=np.float32)
    rgb = utils.ternary_alpha_rgb(v, v, v)
    assert rgb.shape == (2, 2, 3)
    assert np.allclose(rgb[1, 1], np.array(config.OCEAN_RGB), atol=1e-3)
    assert rgb[np.isfinite(rgb)].min() >= 0.0 and rgb.max() <= 1.0
