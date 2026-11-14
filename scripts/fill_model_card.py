"""Fill docs/MODEL_CARD.md from reports/ (uses real outputs only; missing values stay as ${...})."""
import json
from pathlib import Path
from string import Template

from _common import parse


def main():
    _, cfg = parse(__doc__)
    root = Path(__file__).resolve().parents[1]
    rep = Path(cfg["paths"]["reports"])
    vals = {
        "ARCH": cfg["model"]["arch"], "INDICES": ", ".join(cfg["preprocess"]["indices"]),
        "REGION": cfg["region"]["name"], "PERIOD": f"{cfg['period']['start_date']} to {cfg['period']['end_date']}",
        "SPLIT_STRATEGY": cfg["split"]["strategy"],
    }
    if (rep / "summary.json").exists():
        vals["BEST_MODEL"] = json.loads((rep / "summary.json").read_text())["best_model"]
    if (rep / "metrics_table.md").exists():
        vals["METRICS_TABLE"] = (rep / "metrics_table.md").read_text()
    if (rep / "dataset" / "dataset_summary.json").exists():
        d = json.loads((rep / "dataset" / "dataset_summary.json").read_text())
        vals.update(N_TOTAL=d["n_total"], N_POS=d["n_pos"], N_NEG=d["n_neg"], LEAK_PASSED=d["leakage"]["passed"])
    text = Template((root / "docs" / "MODEL_CARD.template.md").read_text()).safe_substitute(vals)
    (root / "docs" / "MODEL_CARD.md").write_text(text)
    print("wrote docs/MODEL_CARD.md")


if __name__ == "__main__":
    main()
