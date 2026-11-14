CONFIG ?= configs/default.yaml
PY ?= python

.PHONY: install install-deep install-demo test plot-labels matched download-labels download-imagery build-dataset report baselines train evaluate demo-assets demo all

install:                ## no torch needed for labels, imagery, dataset, baselines, evaluation
	$(PY) -m pip install -e ".[dev]"

install-deep:           ## adds torch + torchvision (CNN training) ; install-demo also adds streamlit
	$(PY) -m pip install -e ".[deep]"

install-demo:
	$(PY) -m pip install -e ".[demo]"

test:
	$(PY) -m pytest -q

download-labels:        ## needs FIRMS_MAP_KEY in env
	$(PY) scripts/download_labels.py --config $(CONFIG)

download-imagery:       ## STAC match + windowed Sentinel-2 patch extraction
	$(PY) scripts/download_imagery.py --config $(CONFIG)

plot-labels:           ## charts + map of the downloaded FIRMS labels
	$(PY) scripts/plot_labels.py --config $(CONFIG)

build-dataset:          ## QC filters, splits, leakage purge, normalisation stats
	$(PY) scripts/build_dataset.py --config $(CONFIG)

report:                 ## pre-training dataset report and leakage check
	$(PY) scripts/report_dataset.py --config $(CONFIG)

baselines:
	$(PY) scripts/run_baselines.py --config $(CONFIG)

train:
	$(PY) scripts/train.py --config $(CONFIG)

evaluate:
	$(PY) scripts/evaluate.py --config $(CONFIG)

matched:                ## same pipeline on the land-cover-matched dataset (no PyTorch needed)
	$(PY) scripts/build_dataset.py --config configs/matched.yaml
	$(PY) scripts/report_dataset.py --config configs/matched.yaml
	$(PY) scripts/run_baselines.py --config configs/matched.yaml
	$(PY) scripts/evaluate.py --config configs/matched.yaml

demo-assets:
	$(PY) scripts/export_demo_assets.py --config $(CONFIG)

demo:
	streamlit run app/streamlit_app.py

all: download-labels download-imagery build-dataset report baselines train evaluate
