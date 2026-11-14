"""Evaluate heuristic, classical ML, and CNN on the test split; write tables and figures."""
import json

from _common import parse

from wildfire.evaluation import evaluate_all


def extra(ap):
    ap.add_argument("--run", default=None, help="run dir containing best.pt (default: latest)")


def main():
    args, cfg = parse(__doc__, extra)
    s = evaluate_all(cfg, args.run)
    print(json.dumps({k: v for k, v in s.items() if k != "models"}, indent=2, default=str))
    print("see reports/metrics_table.md")


if __name__ == "__main__":
    main()
