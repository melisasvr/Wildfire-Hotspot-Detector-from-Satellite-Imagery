"""Shared CLI helpers for pipeline scripts."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:  # read FIRMS_MAP_KEY from <repo root>/.env
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True, encoding="utf-8-sig")
except ImportError:
    pass

from wildfire.config import DEFAULT_CONFIG, load_config  # noqa: E402


def parse(description, extra=None):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    if extra:
        extra(ap)
    args = ap.parse_args()
    return args, load_config(args.config)