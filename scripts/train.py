"""Train the CNN on the train split, early-stop on validation PR-AUC."""
import sys

from _common import parse

try:
    import torch  # noqa: F401
except ImportError:
    sys.exit("PyTorch is not installed. Run `make install-deep`, or train the CNN on Colab/Kaggle. "
             "The baselines and evaluation work without it.")

from wildfire.train import train


def main():
    _, cfg = parse(__doc__)
    print("run dir:", train(cfg))


if __name__ == "__main__":
    main()
