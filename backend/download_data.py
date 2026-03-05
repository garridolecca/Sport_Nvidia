"""Download the Alfheim Stadium dataset (video + ZXY tracking ground truth)."""

import os
import requests
from tqdm import tqdm
from config import DATA_DIR


ALFHEIM_BASE = "https://datasets.simula.no/downloads/alfheim"

# Dataset files — video clips and tracking CSVs
FILES = [
    # Two camera-array video files (left + right halves of the pitch)
    "2013-11-03/alfheim-2013-11-03_cam1.avi",
    "2013-11-03/alfheim-2013-11-03_cam2.avi",
    # ZXY ground-truth player tracking data
    "2013-11-03/alfheim-2013-11-03_zxy.csv",
]


def download_file(url: str, dest: str):
    """Download a file with a progress bar."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return

    print(f"  Downloading: {url}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def main():
    print("Downloading Alfheim Stadium dataset...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for rel_path in FILES:
        url = f"{ALFHEIM_BASE}/{rel_path}"
        dest = DATA_DIR / rel_path
        try:
            download_file(url, str(dest))
        except Exception as e:
            print(f"  WARNING: Could not download {rel_path}: {e}")
            print("  You may need to download manually from https://datasets.simula.no/alfheim/")

    print("\nDone. Data saved to:", DATA_DIR)
    print("\nIf automatic download fails, visit https://datasets.simula.no/alfheim/")
    print("and place files under:", DATA_DIR / "2013-11-03/")


if __name__ == "__main__":
    main()
