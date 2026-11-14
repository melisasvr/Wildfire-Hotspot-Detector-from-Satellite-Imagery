# Wildfire Hotspot Detector from Satellite Imagery

A patch-level active-fire detector for Sentinel-2 imagery, trained on NASA FIRMS VIIRS detections as **weak labels** and compared honestly against a spectral rule and a classical ML baseline.

<!-- Replace with a GIF or screenshot of the demo: docs/demo.gif -->
![demo](docs/demo.gif)

> **Scope.** This is an active-fire/hotspot detector trained on FIRMS-derived weak labels. It does not detect "all wildfires", and it is not an early-warning or emergency-response system.

## Why this problem is interesting

FIRMS gives near-global thermal-anomaly detections at 375 m, but they come from a thermal sensor, are noisy, and say nothing about what the fire looks like in optical imagery. This project asks a narrower, testable question: *can a model recognise the optical/SWIR signature of a FIRMS-style detection in a Sentinel-2 patch, and does deep learning add anything over a two-parameter spectral rule?*

## Data

| Role | Source | Notes |
|---|---|---|
| Labels | [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) VIIRS 375 m (SNPP and NOAA-20, standard-processing archive) | Open NASA data. Confidence, FRP, time, day/night kept. |
| Imagery | Copernicus Sentinel-2 L2A via the [Earth Search STAC API](https://element84.com/earth-search/) (public COGs) | Free and open (Copernicus). Same mission data as the Copernicus Data Space Ecosystem, without the credentials step. |

Default study area: Northern California (`configs/default.yaml`), June to October 2024. Chosen for frequent summer fires and dense Sentinel-2 coverage; change `region` and `period` to use anything else.

FIRMS thermal bands are **never** model inputs, which would leak the label signal.

## Method

```mermaid
flowchart LR
  A[FIRMS VIIRS archive] --> B[Filter: confidence, FRP, persistent sources]
  B --> C["STAC match: same scene, |time gap| <= 6 h, low cloud"]
  D --> E[Positives centred on detections;<br/>negatives from the same scene]
  E --> F[Spatial-block split + leakage purge]
  F --> G[Train-only normalisation + indices]
  G --> H1[Spectral rule]
  G --> H2[Random forest]
  G --> H3[ResNet-18 / small CNN]
  H1 & H2 & H3 --> I[Test evaluation, slices, error analysis]
  H3 --> J[Streamlit demo]

Key design decisions (and the reasons):

- **Bands:** B02, B03, B04, B08 at 10 m, B11 and B12 at 20 m upsampled to 10 m with nearest-neighbour. Bilinear blurs a single hot pixel and roughly halves its peak, so it was rejected.
- **Patches:** 128 x 128 px (1.28 km), corners snapped to the Sentinel-2 grid, centred on the detection within 10 m.
- **Time matching:** Sentinel-2 passes around 10:30 local; VIIRS afternoon passes are about 3 h later, and night passes are 9+ h away. The time gap is stored for every patch and analysed as an error source. Night-time detections are effectively excluded by the 6 h limit.
- **Negatives:** drawn from the same scene as their positive, 3 to 20 km away, at least 2 km from *any* FIRMS detection (all confidence levels, plus or minus one day). "No detection nearby" is not proof of "no fire", so negatives are weak too.
- **Splits:** whole 25 km spatial blocks go to train/val/test; any cross-split pair closer than 3 km is purged from the lower-priority side. A leakage report must pass before training.
- **Normalisation:** statistics are computed on the training split only.
- **Imbalance:** class-weighted BCE (or focal loss); augmentation limited to flips and 90-degree rotations so spectral values stay untouched.
- **Precision depends on the sampled ratio.** The dataset is not real-world prevalence, so the evaluation also reports precision at assumed prevalences. Treat those as illustrations only.

## How to run it, step by step

You need Python 3.10 to 3.12 and an internet connection. Run every command from the **project root folder** (the one containing `pyproject.toml`), because the scripts look for `configs/default.yaml` relative to it.

### 1. Set up the environment

```bash
cd wildfire-hotspot-detector
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"            # everything except PyTorch
python -m pytest -q                # sanity check: should pass (torch tests are skipped without torch)
```

PyTorch is optional. Install it only for CNN training and the demo: `pip install -e ".[deep]"`. If it fails on your machine, skip it, or train the CNN on Colab or Kaggle. Everything else works without it.

### 2. Add your NASA FIRMS key

Get a free key at https://firms.modaps.eosdis.nasa.gov/api/map_key/ (it arrives by email). Then create a file named `.env` in the project root with one line, no quotes and no spaces:

```
FIRMS_MAP_KEY=your_key_here
```

The scripts load it automatically. (Alternative: set it in your terminal with `export FIRMS_MAP_KEY=...` on Mac/Linux, or `$env:FIRMS_MAP_KEY="..."` in PowerShell.)

### 3. Make a small first run

Edit `configs/default.yaml` before downloading anything. A small box and a short period keep the first run fast and cheap. For example, around the July 2024 Park Fire:

```yaml
region:
  bbox: [-122.4, 39.6, -121.0, 40.6]
period:
  start_date: "2024-07-24"
  end_date: "2024-08-10"
sampling:
  max_positives: 200
```

Scale up once the whole pipeline works.

### 4. Run the pipeline in order

| Step | Command with make | Command without make (Windows) | What it does |
|---|---|---|---|
| 1 | `make download-labels` | `python scripts/download_labels.py` | Downloads FIRMS detections, filters them, prints how many each filter removed |
| 2 | `make download-imagery` | `python scripts/download_imagery.py` | Finds matching Sentinel-2 scenes, reads small patches |
| 3 | `make build-dataset` | `python scripts/build_dataset.py` | Splits by area, purges leaks, computes normalisation |
| 4 | `make report` | `python scripts/report_dataset.py` | Counts, class balance, cloud histogram, example patches |
| 5 | `make baselines` | `python scripts/run_baselines.py` | Fits the spectral rule and random forest |
| 6 | `make train` | `python scripts/train.py` | Trains the CNN (needs PyTorch) |
| 7 | `make evaluate` | `python scripts/evaluate.py` | Metrics, curves, error grids in `reports/` |
| 8 | `make demo-assets` then `make demo` | `python scripts/export_demo_assets.py` then `streamlit run app/streamlit_app.py` | Streamlit demo (needs PyTorch) |

Stop after step 1 and step 4 and look at the output. Step 1 tells you how many detections survive filtering. Step 4 writes figures to `reports/dataset/`, including the leakage check, and you should read them before training anything.

`make` comes with Mac and Linux. On Windows, use the "without make" column.

### 5. Where things end up

- `data/raw/firms/` downloaded labels
- `data/processed/patches/` extracted patches (not committed to Git)
- `data/processed/manifest.parquet` one row per patch, with split and metadata
- `runs/` baselines and CNN checkpoints
- `reports/` metrics table, figures, error examples

### Troubleshooting

- **"Set FIRMS_MAP_KEY first"**: the `.env` file is in the wrong folder or has a typo. It belongs in the project root.
- **"No matches"**: widen the dates, raise `imagery.max_time_gap_hours` or `imagery.max_scene_cloud`. Same-day Sentinel-2 coverage is limited.
- **ModuleNotFoundError: wildfire**: run `pip install -e ".[dev]"` again from the project root with the virtual environment active.
- **Leakage check failed**: read `data/processed/leakage_report.json`. Usually there is too little data for the split settings.

## Fairer evaluation: matched negatives and confidence intervals

**Matched negatives.** Random negatives from the same scene include many easy ones (farmland, towns), while positives sit in wildland, so a model can look good just by telling landscapes apart. `configs/matched.yaml` keeps every positive and, within each split, keeps only the negatives whose surroundings look most like a positive's (median NDVI and median B11, within a caliper). It reuses the already-downloaded patches and writes to separate folders:

```bash
python scripts/build_dataset.py --config configs/matched.yaml   # prints balance before and after matching
python scripts/report_dataset.py --config configs/matched.yaml
python scripts/run_baselines.py --config configs/matched.yaml
python scripts/evaluate.py --config configs/matched.yaml        # results in reports_matched/
```

Compare `reports/` and `reports_matched/`: the drop in precision is how much the easy negatives were flattering the model. Treat the matched set as a harder stress test, not as "the truth", because burned ground near a fire also shifts these features.

**Confidence intervals.** `evaluate.py` also writes `metrics_ci.csv`, `metrics_table_ci.md` and `model_differences.csv`. Intervals come from resampling whole 25 km spatial blocks (patches near each other share fires, so they are not independent). `model_differences.csv` uses the same resamples for every model; if the interval for "A minus B" includes 0, the test set cannot separate those models. Intervals cover test-set noise only, not training randomness or the validation-chosen threshold.

## Results

- Run `make evaluate`, then paste `reports/metrics_table.md` here. Nothing in this repository ships with pre-filled numbers.
- Evaluated on the test split (`n = 90` patches, `n_pos = 28` positives):

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | ECE | Brier |
|---|---|---|---|---|---|---|---|
| Spectral rule | 0.870 | 0.714 | 0.784 | 0.842 | 0.882 | 0.098 | 0.093 |
| Classical ML | 0.591 | 0.929 | 0.722 | 0.931 | 0.927 | 0.128 | 0.084 |
| CNN (resnet18) | 0.909 | 0.714 | 0.800 | 0.891 | 0.897 | 0.128 | 0.104 |

- Also produced: `pr_curves.png`, `reliability.png`, confusion matrices, `slices_*.csv` (FIRMS confidence, FRP, time gap, cloud), `false_positives.png`, `false_negatives.png`, `prediction_map.png/.html`.

## Error analysis and limitations

- Fill this in from the figures after evaluation. Things to check by eye:

- **False positives:** industrial heat, solar farms, bare bright soil, burn scars, cloud edges, water glare, registration offsets.
- **False negatives:** small or low-FRP fires, smoke or cloud over the fire, large FIRMS/Sentinel-2 time gaps.

- Known limits that hold regardless of results:

- FIRMS is weak supervision; the "ground truth" has its own false alarms and misses.
- Sentinel-2 revisit time, daylight-only imaging, and clouds cap how often a detection can even be observed.
- Persistent-source filtering is a crude proxy and can remove real long-burning fires.
- Patch classification with centred positives uses a location prior; scanning whole scenes would need a sliding window or a segmentation head.
- Not for emergency response or safety decisions.

## Repository layout

```
configs/    YAML config (region, dates, bands, patch size, model, training)
src/wildfire/  firms, stac, patches, extract, dataset, preprocess, baselines, models, train, evaluation, infer
scripts/    one CLI entry point per stage
app/        Streamlit demo
tests/      coordinate conversion, patch extraction and alignment, negative rules, splits and leakage, loaders
docs/       model card template, LinkedIn post draft
```

## Future work

Pixel-level segmentation head; pre-fire plus during-fire change input (compare against single-date); temporal-gap-aware training; land-cover slices with ESA WorldCover; sliding-window scene-level inference with a proper alarm-rate analysis.

## License

MIT for the code. FIRMS data: NASA open data policy. Sentinel-2: Copernicus free, full and open data licence.
