import numpy as np

from conftest import add_hot_pixel
from wildfire.geo import xy_to_lonlat
from wildfire.patches import SceneReader, from_uint16, to_uint16, patch_qc


def test_uint16_roundtrip():
    r = np.array([0.0, 0.1234, 0.9999, 1.2], dtype=np.float32)
    assert np.allclose(from_uint16(to_uint16(r)), r, atol=1e-4)


def test_scene_reader_shape_and_hot_pixel_is_centred(fake_scene):
    cx, cy = fake_scene["center"]
    hx, hy = cx + 410.0, cy - 270.0
    add_hot_pixel(fake_scene, hx, hy)
    lon, lat = xy_to_lonlat(hx, hy, fake_scene["crs"])
    with SceneReader(fake_scene["scene"], ["B02", "B03", "B04", "B08", "B11", "B12"], 64, 10.0) as r:
        out = r.read(float(lon), float(lat))
    assert out["refl"].shape == (6, 64, 64)
    b12 = out["refl"][5]
    row, col = np.unravel_index(np.argmax(b12), b12.shape)
    assert abs(row - 32) <= 2 and abs(col - 32) <= 2          # label/pixel alignment
    assert b12.max() > 0.5 and np.isclose(out["refl"][0].mean(), 0.1, atol=1e-3)
    assert out["qc"]["nodata_frac"] == 0 and out["qc"]["cloud_frac"] == 0


def test_window_outside_scene_returns_none(fake_scene):
    cx, cy = fake_scene["center"]
    lon, lat = xy_to_lonlat(cx + 2950.0, cy, fake_scene["crs"])   # patch would hang off the edge
    with SceneReader(fake_scene["scene"], ["B02", "B03", "B04", "B08", "B11", "B12"], 64, 10.0) as r:
        assert r.read(float(lon), float(lat)) is None


def test_patch_qc_cloud_and_water():
    dn = np.ones((6, 4, 4), dtype=np.float32)
    scl = np.full((4, 4), 4, dtype=np.uint8); scl[:2] = 9; scl[3, :] = 6
    qc = patch_qc(dn, scl)
    assert qc["cloud_frac"] == 0.5 and qc["water_frac"] == 0.25
