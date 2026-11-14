import numpy as np

from wildfire.geo import bounds_center, lonlat_to_xy, patch_bounds, xy_to_lonlat


def test_lonlat_xy_roundtrip():
    lon, lat = np.array([-122.3, -121.0]), np.array([39.5, 40.8])
    x, y = lonlat_to_xy(lon, lat, "EPSG:32610")
    lon2, lat2 = xy_to_lonlat(x, y, "EPSG:32610")
    assert np.allclose(lon, lon2, atol=1e-8) and np.allclose(lat, lat2, atol=1e-8)


def test_known_utm_value():
    # central meridian of UTM zone 10N is -123; easting there is 500000 m
    x, _ = lonlat_to_xy(-123.0, 40.0, "EPSG:32610")
    assert abs(float(x) - 500000.0) < 1e-3


def test_patch_bounds_size_center_and_grid():
    b = patch_bounds(512345.0, 4412345.0, 128, 10.0, 20.0)
    assert b[2] - b[0] == 1280 and b[3] - b[1] == 1280
    cx, cy = bounds_center(b)
    assert abs(cx - 512345.0) <= 10 and abs(cy - 4412345.0) <= 10
    assert b[0] % 20 == 0 and b[1] % 20 == 0
