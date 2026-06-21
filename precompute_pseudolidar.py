"""
Precompute and cache the pseudo-LiDAR point cloud for every frame (one-time, offline).

The monocular-depth network is **frozen**, so the pseudo-LiDAR it produces is identical
on every epoch — recomputing it each training step is pure waste (and the dominant cost
of the PseudoPillars stage). This script runs depth once per frame, back-projects to the
camera-frame pseudo-LiDAR, and saves the valid points to ``data.pseudolidar_cache_dir``.
Training then loads these instead of running depth (see PseudoBatch.pseudo_pcl).

Usage:
    python precompute_pseudolidar.py data=kitti_pillars model=pseudopillars \
        depth=depth_anything data_dir=/path/to/kitti_raw \
        data.pseudolidar_cache_dir=/path/to/kitti_raw/pseudo_cache

Safe to re-run: frames already cached are skipped (resume support).
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from pseudocal.pseudolidar import backproject

torch.set_float32_matmul_precision("high")


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cache_dir = cfg.data.get("pseudolidar_cache_dir", None)
    assert cache_dir, "Set data.pseudolidar_cache_dir to the output directory."
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datamodule = instantiate(cfg.data)
    datamodule.setup()
    depth = instantiate(cfg.depth).to(device).eval()

    # Back-projection parameters (kept identical to the training-time live path).
    m = cfg.model
    kw = dict(
        stride=int(m.get("backproject_stride", 2)),
        min_depth=float(m.get("min_depth", 1.0)),
        max_depth=float(m.get("max_depth", 80.0)),
        max_points=int(m.get("max_pseudo_points", 30000)),
    )
    batch_size = int(cfg.data.get("batch_size", 8))

    done = skipped = 0
    for ds in (datamodule.train_ds, datamodule.val_ds, datamodule.test_ds):
        todo = []
        for i in range(len(ds)):
            (todo.append(i) if not ds.cache_path(i).exists() else None)
        skipped += len(ds) - len(todo)

        for j in range(0, len(todo), batch_size):
            chunk = todo[j : j + batch_size]
            samples = [ds[i] for i in chunk]  # live path → image/edge present
            image = torch.stack([s["image"] for s in samples]).to(device)
            edge = torch.stack([s["edge_mask"] for s in samples]).to(device)
            K = torch.stack([s["K"] for s in samples]).to(device)

            with torch.no_grad():
                depth_map = depth.estimate(image).float()  # (b, 1, H, W)
                cloud = backproject(depth_map, K.float(), edge, **kw)  # (b, P, 4)

            for si, idx in enumerate(chunk):
                pts = cloud[si]
                pts = pts[pts[:, 3] > 0, :3].cpu().numpy().astype(np.float16)  # (Ni, 3)
                np.save(ds.cache_path(idx), pts)
                done += 1

            if (j // batch_size) % 25 == 0:
                print(f"[precompute] {done} cached, {skipped} already present …", flush=True)

    print(
        f"[precompute] DONE: {done} frames cached ({skipped} already present) → {cache_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
