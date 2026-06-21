"""
KITTI dataset for the PseudoPillars stage.

Unlike UniCal++ — which projects the LiDAR into the image plane and feeds an
image + sparse depth map — the PseudoPillars stage needs the **raw RGB image**
(to run monocular depth) and the **raw LiDAR scan** (to build BEV pillars in the
camera frame). We therefore subclass :class:`unical.data.dataset.KittiDataset` to
reuse its sample indexing and calibration parsing, but return a different sample
payload and collate into a :class:`PseudoBatch`.

The regression target and the ``metadata`` (T_init / T_gt / T_decal / K) keep the
exact names UniCal's losses and metrics expect, so those components are reused
unchanged by the PseudoPillars LightningModule.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np
import torch
from unical.data.dataset import KittiDataset, Split
from unical.data.decalibrator import ErrorGenerator, ErrorGenerator6D
from unical.utils.geometry import imagenet_normalize, load_image_rgb
from unical.utils.transform import Transform

from pseudocal.pseudolidar import canny_edge_mask


def prepare_camera_input(
    image_rgb: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    edge_low: int = 50,
    edge_high: int = 150,
    edge_dilate: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Resize an RGB image to the working resolution, scale its intrinsics to match,
    compute the Canny keep-mask, and ImageNet-normalise.

    Shared by :class:`PseudoKittiDataset` and the cascade runner so the camera
    pre-processing is identical at train and inference time.

    Args:
        image_rgb: (H0, W0, 3) uint8 RGB image.
        K:         (3, 3) intrinsics for the original resolution.

    Returns:
        ``(image (3, H, W) float tensor, edge_mask (1, H, W) bool tensor,
        K_scaled (3, 3) float32 ndarray)``.
    """
    orig_h, orig_w = image_rgb.shape[:2]
    img = cv2.resize(image_rgb, (width, height))
    K_scaled = K.astype(np.float32).copy()
    K_scaled[0] *= width / orig_w
    K_scaled[1] *= height / orig_h

    edge_keep = canny_edge_mask(img, low=edge_low, high=edge_high, dilate=edge_dilate)
    img_n = imagenet_normalize(img).astype(np.float32)

    return (
        torch.from_numpy(img_n).permute(2, 0, 1),
        torch.from_numpy(edge_keep)[None],
        K_scaled,
    )


class PseudoBatch(NamedTuple):
    """
    Inputs fed to the PseudoPillars model.

    pcl:        (B, N, 4)      — raw padded LiDAR scan [x, y, z, intensity]
    K:          (B, 3, 3)      — camera intrinsics, scaled to (H, W)
    target_reg: tuple[(B, 3), (B, 3, 3)] — (translation, rotation_matrix) decal target
    metadata:   list of per-sample dicts (T_gt, T_init, T_decal, K, img_name)
    image:      (B, 3, H, W)   — ImageNet-normalised RGB (input to monocular depth);
                                 ``None`` when pre-computed pseudo-LiDAR is cached.
    edge_mask:  (B, 1, H, W)   — bool keep-mask (False on depth-discontinuity edges)
    pseudo_pcl: (B, P, 3)      — *cached* camera-frame pseudo-LiDAR points (padded with
                                 zero rows). When present, the model skips the (frozen,
                                 expensive) monocular-depth forward and uses these
                                 directly. ``None`` falls back to computing depth live.
    """

    pcl: torch.Tensor
    K: torch.Tensor
    target_reg: tuple[torch.Tensor, torch.Tensor]
    metadata: list[dict[str, Any]]
    image: torch.Tensor | None = None
    edge_mask: torch.Tensor | None = None
    pseudo_pcl: torch.Tensor | None = None

    # Convenience alias so code/loops that probe ``batch.img`` (à la UniCal) work.
    @property
    def img(self) -> torch.Tensor | None:
        return self.image


class PseudoKittiDataset(KittiDataset):
    """
    KITTI raw loader producing (RGB image, edge mask, raw LiDAR, decal target).

    Args:
        data_dir:      Root of the KITTI raw dataset.
        split:         List of ``(date_str, [drive_ids])`` tuples.
        decalibrator:  Random decalibration generator (per-axis or scalar).
        width, height: Working image resolution (also the depth-map resolution).
        edge_low/high/dilate: Canny parameters for the pseudo-LiDAR keep mask.
        deterministic: Seed the decalibration per-index (used for val/test).
        seed:          Base seed for deterministic decalibration.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: Split,
        decalibrator: ErrorGenerator | ErrorGenerator6D,
        width: int = 512,
        height: int = 512,
        edge_low: int = 50,
        edge_high: int = 150,
        edge_dilate: int = 2,
        pseudolidar_cache_dir: str | Path | None = None,
        deterministic: bool = False,
        seed: int = 0,
    ) -> None:
        # NB: we intentionally do not call ``super().__init__`` — it requires a
        # UniCal preprocessor we do not use. We reuse its parsing methods instead.
        self.data_dir = Path(data_dir)
        self.decalibrator = decalibrator
        self.deterministic = deterministic
        self.seed = seed
        self.width = width
        self.height = height
        self.edge_low = edge_low
        self.edge_high = edge_high
        self.edge_dilate = edge_dilate
        # When set, per-frame pseudo-LiDAR is loaded from this directory instead of
        # re-running monocular depth every step (see scripts/precompute_pseudolidar.py).
        self.pseudolidar_cache_dir = Path(pseudolidar_cache_dir) if pseudolidar_cache_dir else None

        self._samples: list[tuple[str, int, int]] = []
        self._date_meta: dict[str, dict] = {}
        self._parse(split)  # inherited from KittiDataset

    # ------------------------------------------------------------------
    # Decalibration sampling
    # ------------------------------------------------------------------

    def _sample_decalib(self, idx: int) -> Transform:
        """
        Sample the decalibration for sample ``idx``.

        For val/test (``deterministic=True``) the sample is reproducible per index.
        UniCal's scalar :class:`ErrorGenerator` accepts a seeded ``generator``; the
        per-axis :class:`ErrorGenerator6D` does not, so we fall back to seeding the
        global RNG — which makes its ``uniform_`` calls deterministic too.
        """
        if not self.deterministic:
            return self.decalibrator()
        seed = self.seed * 1_000_003 + idx
        try:
            return self.decalibrator(generator=torch.Generator().manual_seed(seed))
        except TypeError:
            torch.manual_seed(seed)
            return self.decalibrator()

    # ------------------------------------------------------------------
    # Pseudo-LiDAR cache
    # ------------------------------------------------------------------

    def cache_path(self, idx: int) -> Path:
        """Path of the cached pseudo-LiDAR points (.npy) for sample ``idx``."""
        date, drive, fid = self._samples[idx]
        assert self.pseudolidar_cache_dir is not None
        return self.pseudolidar_cache_dir / f"{date}_drive{drive:04d}_{fid:010d}.npy"

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        date, drive, fid = self._samples[idx]
        meta = self._date_meta[date]
        K = meta["K"].copy()
        T_gt = meta["T_gt"]

        drive_dir = self.data_dir / date / f"{date}_drive_{drive:04d}_sync"
        lidar_path = drive_dir / "velodyne_points" / "data" / f"{fid:010d}.bin"
        raw_pcl = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 4)

        sample: dict = {
            "pcl": torch.from_numpy(raw_pcl),  # (N, 4)
        }

        cached = self.pseudolidar_cache_dir is not None and self.cache_path(idx).exists()
        if cached:
            # Fast path: load pre-computed pseudo-LiDAR; skip the image + Canny + the
            # (frozen, expensive) monocular-depth forward entirely.
            pts = np.load(self.cache_path(idx)).astype(np.float32)  # (P, 3)
            sample["pseudo_pcl"] = torch.from_numpy(pts)
        else:
            img = load_image_rgb(str(drive_dir / "image_02" / "data" / f"{fid:010d}.png"))
            image, edge_mask, K = prepare_camera_input(
                img,
                K,
                self.width,
                self.height,
                edge_low=self.edge_low,
                edge_high=self.edge_high,
                edge_dilate=self.edge_dilate,
            )
            sample["image"] = image  # (3, H, W)
            sample["edge_mask"] = edge_mask  # (1, H, W)

        # Sample the decalibration to predict (deterministic for val/test).
        T_decal = self._sample_decalib(idx)
        T_init = T_decal @ T_gt

        sample.update(
            {
                "K": torch.from_numpy(K.astype(np.float32)),  # (3, 3)
                "trans": torch.from_numpy(T_decal.translation),  # (3,)
                "rot_mat": torch.from_numpy(T_decal.rotation_matrix),  # (3, 3)
                "metadata": {
                    "T_gt": T_gt,
                    "T_init": T_init,
                    "T_decal": T_decal,
                    "K": K,
                    "img_name": f"{fid:010d}",
                },
            }
        )
        return sample

    # ------------------------------------------------------------------
    # Raw access (used by the cascade runner)
    # ------------------------------------------------------------------

    def raw_sample(self, idx: int) -> dict:
        """
        Return the *un-preprocessed* ingredients for one sample: original RGB image,
        raw LiDAR, original intrinsics, ground-truth and decalibrated extrinsics.

        The cascade renders each stage's inputs itself, so it needs the raw data plus
        the (deterministic) decalibration this dataset would apply.
        """
        date, drive, fid = self._samples[idx]
        meta = self._date_meta[date]
        drive_dir = self.data_dir / date / f"{date}_drive_{drive:04d}_sync"
        img = load_image_rgb(str(drive_dir / "image_02" / "data" / f"{fid:010d}.png"))
        raw_pcl = np.fromfile(
            str(drive_dir / "velodyne_points" / "data" / f"{fid:010d}.bin"),
            dtype=np.float32,
        ).reshape(-1, 4)

        T_decal = self._sample_decalib(idx)
        T_gt = meta["T_gt"]
        return {
            "image_rgb": img,
            "pcl": raw_pcl,
            "K": meta["K"].copy(),
            "T_gt": T_gt,
            "T_init": T_decal @ T_gt,
        }

    # ------------------------------------------------------------------
    # Collate
    # ------------------------------------------------------------------

    @staticmethod
    def collate(samples: list[dict]) -> PseudoBatch:
        K = torch.stack([s["K"] for s in samples])
        trans = torch.stack([s["trans"] for s in samples])
        rot_mat = torch.stack([s["rot_mat"] for s in samples])
        pcl = torch.nn.utils.rnn.pad_sequence([s["pcl"] for s in samples], batch_first=True)
        metadata = [s["metadata"] for s in samples]

        # Optional fields: present together depending on the cached vs live path.
        has_img = "image" in samples[0]
        image = torch.stack([s["image"] for s in samples]) if has_img else None
        edge_mask = torch.stack([s["edge_mask"] for s in samples]) if has_img else None
        pseudo_pcl = None
        if "pseudo_pcl" in samples[0]:
            pseudo_pcl = torch.nn.utils.rnn.pad_sequence(
                [s["pseudo_pcl"] for s in samples], batch_first=True
            )

        return PseudoBatch(
            pcl=pcl,
            K=K,
            target_reg=(trans, rot_mat),
            metadata=metadata,
            image=image,
            edge_mask=edge_mask,
            pseudo_pcl=pseudo_pcl,
        )
