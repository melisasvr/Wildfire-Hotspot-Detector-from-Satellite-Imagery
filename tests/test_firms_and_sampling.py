import numpy as np
import pandas as pd

from wildfire.firms import filter_detections, flag_persistent_sources, parse_detections
from wildfire.sampling import far_from, propose_negative_centres, thin_points
from scipy.spatial import cKDTree


def _raw():
    rows = []
    for i in range(30):   # a static heat source: same pixel, 30 distinct days
        rows.append(dict(latitude=40.1, longitude=-121.1, acq_date=f"2024-07-{(i % 28) + 1:02d}", acq_time=2130,
                         confidence="n", frp=12.0, daynight="D", satellite="N", source="S"))
    rows += [
        dict(latitude=40.5, longitude=-121.5, acq_date="2024-07-02", acq_time=2130, confidence="h", frp=30.0, daynight="D", satellite="N", source="S"),
        dict(latitude=40.6, longitude=-121.6, acq_date="2024-07-02", acq_time=130, confidence="l", frp=3.0, daynight="N", satellite="N", source="S"),
        dict(latitude=40.7, longitude=-121.7, acq_date="2024-07-03", acq_time=2131, confidence="n", frp=0.0, daynight="D", satellite="N", source="S"),
    ]
    return pd.DataFrame(rows)


def test_parse_time_and_confidence():
    df = parse_detections(_raw())
    assert df.acq_datetime.iloc[-3] == pd.Timestamp("2024-07-02 21:30", tz="UTC")
    assert df.acq_datetime.iloc[-2] == pd.Timestamp("2024-07-02 01:30", tz="UTC")
    assert list(df.conf_ord.iloc[-3:]) == [2, 0, 1]


def test_filters_are_logged_and_drop_the_right_rows():
    df = parse_detections(_raw())
    f, log = filter_detections(df, {"min_confidence": "nominal", "min_frp_mw": 0.0, "drop_persistent_sources": True,
                                    "persistent_cell_deg": 0.005, "persistent_min_days": 8}, [-123.5, 38.8, -120.5, 41.8])
    assert len(f) == 1 and f.confidence.iloc[0] == "h"        # low conf, zero FRP, persistent source removed
    assert [r["step"] for r in log][0] == "start" and log[-1]["n"] == 1
    assert flag_persistent_sources(df, 0.005, 8).iloc[:30].all()


def test_thinning_and_negative_proposals():
    xy = np.array([[0, 0], [100, 0], [5000, 0], [5050, 0]], dtype=float)
    keep = thin_points(xy, 1000, priority=np.array([1, 2, 3, 4]))
    assert sorted(keep.tolist()) == [1, 3]                   # highest-priority point of each close pair
    rng = np.random.default_rng(0)
    c = propose_negative_centres([0, 0], rng, 500, 3000, 20000)
    d = np.linalg.norm(c, axis=1)
    assert d.min() >= 3000 and d.max() <= 20000
    assert not far_from(np.array([[10.0, 0.0]]), cKDTree(xy), 500)[0]
