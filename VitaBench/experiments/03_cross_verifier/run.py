"""Experiment entry point."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experimentkit.cli import main

if __name__ == "__main__":
    main(HERE / "config.json")
