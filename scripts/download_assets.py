"""Download RoboTwin 2.0 simulation assets at a pinned dataset revision and extract them
into RoboTwin/assets, then absolutize embodiment config paths (RoboTwin's own helper).

background_texture.zip (~11 GB) is skipped: demo_clean has random_background=false."""

import os
import pathlib
import subprocess
import sys
import zipfile

from huggingface_hub import hf_hub_download

REV = "3dc3b798668feb99ac61cc9086d84cbcc3d79186"  # TianxingChen/RoboTwin2.0, see configs/pins.txt
ROOT = pathlib.Path(os.environ.get("ROBOTWIN_ROOT", pathlib.Path(__file__).resolve().parents[1] / "RoboTwin"))


def main():
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    for fname in ("embodiments.zip", "objects.zip"):
        if (assets / fname[:-4]).is_dir():
            print("exists, skipping", fname)
            continue
        p = hf_hub_download("TianxingChen/RoboTwin2.0", fname, repo_type="dataset", revision=REV, local_dir=str(assets))
        with zipfile.ZipFile(p) as z:
            z.extractall(assets)
        pathlib.Path(p).unlink()
        print("extracted", fname)
    subprocess.run([sys.executable, "script/update_embodiment_config_path.py"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
