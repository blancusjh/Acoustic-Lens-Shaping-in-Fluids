"""Run with python -m acoustic_freeform.dual CONFIG --out artifacts/NAME."""

import argparse
from pathlib import Path

from .campaign import run

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("config", type=Path)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
run(args.config, args.out)
