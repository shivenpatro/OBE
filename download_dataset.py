"""
download_dataset.py
===================
Automates acquisition of the xAPI-Educational Mining Dataset.

Strategy (attempted in order):
    1. Kaggle CLI  — fastest if `kaggle` is installed and credentials exist.
    2. UCI direct  — HTTP download from the UCI ML Repository mirror.
    3. Manual      — Prints exact instructions if both automated paths fail.

Usage:
    python download_dataset.py

The script places `StudentPerformance.csv` in the `data/` directory
alongside this file, which is exactly where `data_loader.py` expects it.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import zipfile

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = pathlib.Path(__file__).parent
DATA_DIR   = ROOT / "data"
DEST       = DATA_DIR / "StudentPerformance.csv"

# Kaggle dataset identifier (owner/dataset-name)
KAGGLE_DATASET = "aljarah/xAPI-Edu-Data"
KAGGLE_CSV_NAME = "xAPI-Edu-Data.csv"

# UCI direct link (CSV is hosted here without authentication)
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00346/xAPI-Edu-Data.csv"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _already_exists() -> bool:
    if DEST.exists():
        print(f"[skip]  Dataset already present at: {DEST}")
        return True
    return False


# ---------------------------------------------------------------------------
# Strategy 1: Kaggle CLI
# ---------------------------------------------------------------------------

def _try_kaggle() -> bool:
    """
    Attempt to download via the official Kaggle CLI.
    Requires:
        pip install kaggle
        ~/.kaggle/kaggle.json  (API key from https://www.kaggle.com/settings)
    """
    if shutil.which("kaggle") is None:
        print("[kaggle] CLI not found — skipping.")
        return False

    kaggle_json = pathlib.Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print(f"[kaggle] Credentials not found at {kaggle_json} — skipping.")
        return False

    print(f"[kaggle] Downloading dataset '{KAGGLE_DATASET}' …")
    result = subprocess.run(
        [
            "kaggle", "datasets", "download",
            "--dataset", KAGGLE_DATASET,
            "--path", str(DATA_DIR),
            "--unzip",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"[kaggle] Download failed:\n{result.stderr.strip()}")
        return False

    # Kaggle names the file 'xAPI-Edu-Data.csv' — rename to our canonical name
    downloaded = DATA_DIR / KAGGLE_CSV_NAME
    if downloaded.exists():
        downloaded.rename(DEST)
        print(f"[kaggle] Saved to: {DEST}")
        return True

    # Check for a zip artifact that wasn't unpacked (older kaggle CLI)
    zip_file = DATA_DIR / f"{KAGGLE_DATASET.split('/')[1]}.zip"
    if zip_file.exists():
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(DATA_DIR)
        zip_file.unlink()
        downloaded = DATA_DIR / KAGGLE_CSV_NAME
        if downloaded.exists():
            downloaded.rename(DEST)
            print(f"[kaggle] Extracted and saved to: {DEST}")
            return True

    print("[kaggle] Download appeared to succeed but CSV not found.")
    return False


# ---------------------------------------------------------------------------
# Strategy 2: Direct HTTP download (UCI)
# ---------------------------------------------------------------------------

def _try_uci_http() -> bool:
    """
    Download the CSV directly from the UCI ML Repository.
    Uses the standard library only (no extra dependencies beyond requests).
    Falls back to urllib if requests is unavailable.
    """
    print(f"[uci]   Attempting direct HTTP download from UCI …")
    print(f"        {UCI_URL}")

    try:
        import requests  # preferred — already in requirements.txt

        response = requests.get(UCI_URL, timeout=30, stream=True)
        if response.status_code != 200:
            print(f"[uci]   HTTP {response.status_code} — skipping.")
            return False

        with open(DEST, "wb") as fh:
            for chunk in response.iter_content(chunk_size=8192):
                fh.write(chunk)

        print(f"[uci]   Saved to: {DEST}")
        return True

    except Exception as exc:  # noqa: BLE001
        print(f"[uci]   requests failed ({exc}) — trying urllib …")

    try:
        import urllib.request

        urllib.request.urlretrieve(UCI_URL, DEST)
        print(f"[uci]   Saved to: {DEST}")
        return True

    except Exception as exc:  # noqa: BLE001
        print(f"[uci]   urllib also failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Strategy 3: Manual instructions
# ---------------------------------------------------------------------------

MANUAL_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════╗
║           MANUAL DATASET DOWNLOAD REQUIRED                       ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Automated download failed. Follow ONE of these options:         ║
║                                                                  ║
║  OPTION A — Kaggle (fastest, full dataset)                       ║
║  ─────────────────────────────────────────                       ║
║  1. Go to:  https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data║
║  2. Click "Download" (you need a free Kaggle account).           ║
║  3. Unzip and rename the file to:  StudentPerformance.csv        ║
║  4. Move it to the data/ directory of this project.              ║
║                                                                  ║
║  OPTION B — UCI ML Repository                                    ║
║  ──────────────────────────────                                  ║
║  1. Download directly:                                           ║
║     https://archive.ics.uci.edu/ml/machine-learning-databases/  ║
║     00346/xAPI-Edu-Data.csv                                      ║
║  2. Rename to StudentPerformance.csv                             ║
║  3. Move to the data/ directory.                                 ║
║                                                                  ║
║  OPTION C — Kaggle CLI (automated, future runs)                  ║
║  ─────────────────────────────────────────────                   ║
║  pip install kaggle                                              ║
║  # Add your API key to ~/.kaggle/kaggle.json                     ║
║  # See: https://www.kaggle.com/settings → "Create New Token"    ║
║  python download_dataset.py   # re-run this script              ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""


def _print_manual_instructions() -> None:
    print(MANUAL_INSTRUCTIONS)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def download() -> bool:
    """
    Run the download strategies in priority order.
    Returns True if the dataset is available in data/ after this call.
    """
    _ensure_data_dir()

    if _already_exists():
        return True

    if _try_kaggle():
        return True

    if _try_uci_http():
        return True

    _print_manual_instructions()
    return False


if __name__ == "__main__":
    success = download()
    if success:
        print("\n[ok]    Dataset ready. Run:  python data_loader.py")
    else:
        print("\n[fail]  Please download manually (see instructions above).",
              file=sys.stderr)
        sys.exit(1)
