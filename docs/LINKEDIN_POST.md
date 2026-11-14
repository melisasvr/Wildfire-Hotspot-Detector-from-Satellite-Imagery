# LinkedIn post (draft: replace every [bracket] with your real results)

I built a wildfire hotspot detector from satellite imagery, and the most useful thing I learned was how much of the work is deciding what your labels actually mean.

The setup: NASA FIRMS thermal detections as weak labels, Sentinel-2 optical and SWIR patches as input, a spatial-block train/val/test split with a leakage check, and three models compared honestly: a two-parameter spectral rule, a random forest, and a small CNN.

The hardest part: matching labels to imagery. Sentinel-2 passes around 10:30 am local, the VIIRS afternoon pass comes about 3 hours later, and fires move. Night detections can't be paired with optical imagery at all. I stored the time gap for every patch and analysed it as an error source.

Result: [one sentence with your real numbers, e.g. "the CNN reached PR-AUC X vs Y for the spectral rule on unseen areas"]. [One sentence on where it fails, from your error grids.]

What I'd say to anyone doing something similar: split by place, not by patch; tune thresholds on validation only; and always ask whether the deep model beats a simple rule.

It's a research demo, not an early-warning system. Code, model card and demo: [link]

#MachineLearning #RemoteSensing #ComputerVision #Wildfire #Sentinel2

---

## 30 to 60 second demo video outline

1. (0-5 s) Title card: "Wildfire hotspot detector: Sentinel-2 + NASA FIRMS". One line: weak labels, honest evaluation.
2. (5-15 s) Screen-record the demo: pick an example patch, show true colour next to SWIR false colour, the hot pixel glowing.
3. (15-25 s) Show probability, threshold slider, and the saliency overlay. Say what it is and isn't.
4. (25-40 s) Cut to `reports/`: PR curve with all three models, then the false-positive grid. Name one failure honestly.
5. (40-50 s) Live tab: type a place and date, fetch a patch, predict.
6. (50-60 s) End card: repo link, "not for emergency use".
