"""Project-wide paths and constants.

The raw data location is configurable so the same code runs on the
evaluator trips in phase 2: set PATHSNAP_DATA to the folder that holds
the partition sub-folders (train/, student_test/, evaluator_test/ ...).
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(
    os.environ.get("PATHSNAP_DATA", REPO_ROOT.parent / "Student folder")
).expanduser()

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MAPS_DIR = RESULTS_DIR / "maps"
LOGS_DIR = RESULTS_DIR / "logs"

PARTITIONS = ("train", "student_test")

SEED = 42
