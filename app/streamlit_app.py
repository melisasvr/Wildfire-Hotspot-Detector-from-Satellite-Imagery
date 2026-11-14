"""Interactive demo: pick an example, upload a patch, or fetch a live Sentinel-2 patch.

Run:  make demo-assets && make demo      (or: streamlit run app/streamlit_app.py)
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wildfire.baselines import SpectralRule  # noqa: E402
from wildfire.config import load_config  # noqa: E402
from wildfire.patches import from_uint16  # noqa: E402
from wildfire.preprocess import BANDS  # noqa: E402
from wildfire.viz import swir_false_color, true_color  # noqa: E402

st.set_page_config(page_title="Wildfire Hotspot Detector", layout="wide")
cfg = load_config(ROOT / "configs" / "default.yaml")
ASSETS = ROOT / cfg["paths"]["demo_assets"]
CKPT = ROOT / cfg["demo"]["checkpoint"]

st.title("Wildfire hotspot detector")
st.warning("Research demo. It flags patches that look like FIRMS-style active-fire detections in Sentinel-2. "
           "It is NOT an early-warning or emergency-response tool, and it misses fires under cloud or smoke.")

if not CKPT.exists():
    st.info("No trained model found. Run `make train`, `make evaluate`, then `make demo-assets`.")
    st.stop()

from wildfire.infer import (fetch_patch_live, load_cnn, load_patch_file,  # noqa: E402
                            predict_patch, saliency_map)


@st.cache_resource
def get_model():
    return load_cnn(CKPT)


@st.cache_resource
def get_rule():
    p = ASSETS / "spectral_rule.json"
    return SpectralRule.load(p) if p.exists() else None


model, ck = get_model()
rule = get_rule()
thr = st.sidebar.slider("Decision threshold", 0.0, 1.0, float(ck.get("val_threshold", 0.5)), 0.01,
                        help="Default is the validation-set max-F1 threshold saved with the model.")
st.sidebar.caption(f"Model: {ck['model_cfg'].get('arch')}  |  best epoch {ck.get('epoch')}")


def show(refl: np.ndarray, meta: dict | None = None, latlon=None):
    p = predict_patch(model, ck, refl)
    c1, c2, c3 = st.columns(3)
    c1.metric("CNN probability", f"{p:.2f}", "hotspot-like" if p >= thr else "no hotspot")
    if rule is not None:
        s = rule.score_patch(refl)
        c2.metric("Spectral-rule score", f"{s:.2f}", "above rule threshold" if s >= rule.threshold else "below rule threshold")
    if meta:
        c3.json(meta, expanded=False)

    sal = saliency_map(model, ck, refl)
    rgb = true_color(refl)
    heat = __import__("matplotlib").colormaps["inferno"](sal)[..., :3]
    overlay = np.clip(rgb * (1 - 0.55 * sal[..., None]) + heat * 0.55 * sal[..., None], 0, 1)
    a, b, c = st.columns(3)
    a.image(rgb, caption="True colour (B04, B03, B02)", use_container_width=True)
    b.image(swir_false_color(refl), caption="SWIR false colour (B12, B11, B08): hot pixels glow orange/red",
            use_container_width=True)
    c.image(overlay, caption="Input-gradient saliency overlay (a rough aid, not proof)", use_container_width=True)
    with st.expander("Individual bands"):
        cols = st.columns(6)
        for i, name in enumerate(BANDS):
            v = refl[i]; v = (v - v.min()) / max(v.max() - v.min(), 1e-6)
            cols[i].image(v, caption=name, use_container_width=True)
    if latlon:
        st.map(pd.DataFrame({"lat": [latlon[0]], "lon": [latlon[1]]}), zoom=9)


tab_ex, tab_up, tab_live = st.tabs(["Examples", "Upload a patch", "Live: place + date"])

with tab_ex:
    ex_file = ASSETS / "examples.json"
    if not ex_file.exists():
        st.info("Run `make demo-assets` to export example test patches.")
    else:
        ex = json.loads(ex_file.read_text())
        labels = [f"{e['patch_id']}  (FIRMS-derived label: {'hotspot' if e['label'] else 'no hotspot'})" for e in ex]
        i = st.selectbox("Example test patch", range(len(ex)), format_func=lambda k: labels[k])
        e = ex[i]
        refl = from_uint16(np.load(ASSETS / f"{e['patch_id']}.npz")["refl"])
        st.caption("The label is a weak label derived from FIRMS, not verified ground truth.")
        show(refl, {k: e[k] for k in ("scene_datetime", "firms_confidence", "frp")}, (e["lat"], e["lon"]))

with tab_up:
    st.write("Upload a 6-band patch in order **B02, B03, B04, B08, B11, B12**: `.npy`/`.npz` (uint16 reflectance x 10000 "
             "or float reflectance) or a 6-band GeoTIFF.")
    up = st.file_uploader("Patch file", type=["npy", "npz", "tif", "tiff"])
    if up is not None:
        try:
            show(load_patch_file(up.name, up.read()))
        except Exception as err:
            st.error(str(err))

with tab_live:
    st.write("Fetches a single Sentinel-2 L2A window from the public STAC catalog. Needs internet access.")
    c1, c2, c3 = st.columns(3)
    lat = c1.number_input("Latitude", value=40.05, format="%.4f")
    lon = c2.number_input("Longitude", value=-121.60, format="%.4f")
    d = c3.date_input("Date", value=date(2024, 7, 25), max_value=date.today() - timedelta(days=1))
    if st.button("Fetch and predict"):
        with st.spinner("Searching scenes and reading the patch..."):
            try:
                refl, meta = fetch_patch_live(lat, lon, d.isoformat(), cfg)
                show(refl, meta, (lat, lon))
            except Exception as err:
                st.error(str(err))
