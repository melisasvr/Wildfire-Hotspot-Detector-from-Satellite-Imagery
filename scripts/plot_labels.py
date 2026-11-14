"""Quick look at the downloaded FIRMS labels: charts, a static map, and an interactive map.

Run after `python scripts/download_labels.py`. Outputs go to reports/labels/.
"""
import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from _common import parse

from wildfire.config import path_of


def main():
    _, cfg = parse(__doc__)
    src = path_of(cfg, "raw") / "firms"
    out = path_of(cfg, "reports") / "labels"
    out.mkdir(parents=True, exist_ok=True)
    allp = pd.DataFrame(gpd.read_parquet(src / "firms_all.parquet").drop(columns="geometry"))
    flt = pd.DataFrame(gpd.read_parquet(src / "firms_filtered.parquet").drop(columns="geometry"))
    allp["acq_date"] = pd.to_datetime(allp["acq_date"]); flt["acq_date"] = pd.to_datetime(flt["acq_date"])

    # 1) detections per day, and confidence / FRP summaries
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    allp.groupby("acq_date").size().plot(ax=ax[0], label="all", color="grey")
    flt.groupby("acq_date").size().plot(ax=ax[0], label="after filters", color="C3")
    ax[0].set_title("FIRMS detections per day"); ax[0].set_xlabel(""); ax[0].legend()
    allp["confidence"].value_counts().reindex(["l", "n", "h"]).plot.bar(ax=ax[1], color=["C1", "C0", "C2"])
    ax[1].set_title("Confidence (l / n / h), all detections"); ax[1].set_xlabel("")
    ax[2].hist(flt["frp"].clip(upper=flt["frp"].quantile(0.99)), bins=40, color="C3")
    ax[2].set_title("Fire radiative power (MW), filtered; top 1% clipped"); ax[2].set_xlabel("FRP (MW)")
    fig.tight_layout(); fig.savefig(out / "detections_overview.png", dpi=140); plt.close(fig)

    # 2) static map coloured by FRP
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(flt.longitude, flt.latitude, c=flt.frp.clip(upper=flt.frp.quantile(0.99)), s=3, cmap="inferno_r")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_title(f"Filtered detections (n={len(flt)})")
    fig.colorbar(sc, label="FRP (MW)"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "detections_map.png", dpi=140); plt.close(fig)

    # 3) interactive map you can open in a browser
    try:
        import folium
        d = flt.sample(min(4000, len(flt)), random_state=0)
        m = folium.Map(location=[d.latitude.mean(), d.longitude.mean()], zoom_start=8, tiles="OpenStreetMap")
        cmap = {"h": "red", "n": "orange", "l": "gray"}
        for _, r in d.iterrows():
            folium.CircleMarker([r.latitude, r.longitude], radius=3, color=cmap.get(r.confidence, "blue"),
                                fill=True, fill_opacity=.7,
                                tooltip=f"{r.acq_date:%Y-%m-%d} {int(r.acq_time):04d} UTC, conf {r.confidence}, FRP {r.frp}").add_to(m)
        m.save(out / "detections_map.html")
    except ImportError:
        print("folium not installed, skipped the interactive map")
    print("saved to", out)


if __name__ == "__main__":
    main()