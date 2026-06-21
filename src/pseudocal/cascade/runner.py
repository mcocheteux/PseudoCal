"""
PseudoCal cascade runner.

Chains the three stages at inference time. Each stage observes the calibration as
it currently stands, predicts the *residual* decalibration ``D`` (a rigid
transform), and the running extrinsic estimate is updated by applying the inverse
(the recalibration):

    T_estimate ← D⁻¹ · T_estimate

Starting from a (possibly wildly) decalibrated ``T_init``:

    T₀ = T_init
    T₁ = D₁⁻¹ · T₀     (PseudoPillars — coarse, initialisation-free)
    T₂ = D₂⁻¹ · T₁     (UniCal-M     — medium refinement)
    T₃ = D₃⁻¹ · T₂     (UniCal-S     — fine refinement)

The final ``T₃`` is compared against the ground-truth extrinsic ``T_gt``.

Each stage renders its **own** input representation from the current estimate:
PseudoPillars builds pseudo-LiDAR + real-LiDAR BEV pillars; the UniCal refiners
project the LiDAR into the image plane via UniCal's ``DataPreprocessor``. Stages
therefore share nothing but the running ``Transform`` and the raw sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from unical.data.dataset import Batch as UniCalBatch
from unical.data.preprocessor import DataPreprocessor
from unical.models.module import UniCal
from unical.utils.metrics import CalibMetrics
from unical.utils.transform import Transform, rotation_6d_to_matrix

from pseudocal.data.dataset import prepare_camera_input
from pseudocal.models.pseudopillars import PseudoPillars


@dataclass
class RawSample:
    """Un-preprocessed inputs for one frame (see ``PseudoKittiDataset.raw_sample``)."""

    image_rgb: np.ndarray  # (H0, W0, 3) uint8 RGB
    pcl: np.ndarray  # (N, 4) float32 [x, y, z, intensity]
    K: np.ndarray  # (3, 3) original intrinsics
    T_gt: Transform  # ground-truth LiDAR→camera extrinsic
    T_init: Transform  # decalibrated extrinsic (cascade input)


def _pred_to_transform(pred: tuple[torch.Tensor, torch.Tensor]) -> Transform:
    """Convert a model's ``(trans (1,3), rot6d (1,6))`` output to a ``Transform``."""
    t, r6 = pred
    R = rotation_6d_to_matrix(r6)[0].detach().float().cpu().numpy()
    t = t[0].detach().float().cpu().numpy()
    return Transform.from_rotation_translation(R, t)


class CascadeStage(Protocol):
    """A cascade stage predicts the residual decalibration given the current estimate."""

    def predict(self, raw: RawSample, T_current: Transform, device: torch.device) -> Transform: ...


class PseudoPillarsStage:
    """
    Wraps a trained :class:`PseudoPillars` model as a cascade stage.

    Args:
        model:       Trained PseudoPillars module (set to eval).
        width/height: Working resolution (defaults read from the model's hparams).
        edge_*:      Canny parameters for the pseudo-LiDAR keep mask.
    """

    def __init__(
        self,
        model: PseudoPillars,
        width: int = 512,
        height: int = 512,
        edge_low: int = 50,
        edge_high: int = 150,
        edge_dilate: int = 2,
    ) -> None:
        self.model = model.eval()
        self.width = width
        self.height = height
        self.edge_low = edge_low
        self.edge_high = edge_high
        self.edge_dilate = edge_dilate

    @torch.no_grad()
    def predict(self, raw: RawSample, T_current: Transform, device: torch.device) -> Transform:
        # Import here to avoid a circular import at module load time.
        from pseudocal.data.dataset import PseudoBatch

        image, edge_mask, K = prepare_camera_input(
            raw.image_rgb,
            raw.K,
            self.width,
            self.height,
            edge_low=self.edge_low,
            edge_high=self.edge_high,
            edge_dilate=self.edge_dilate,
        )
        batch = PseudoBatch(
            image=image[None].to(device),
            edge_mask=edge_mask[None].to(device),
            pcl=torch.from_numpy(raw.pcl)[None].to(device),
            K=torch.from_numpy(K)[None].to(device),
            target_reg=(torch.zeros(1, 3, device=device), torch.eye(3, device=device)[None]),
            metadata=[{"T_init": T_current, "T_gt": raw.T_gt, "K": K}],
        )
        return _pred_to_transform(self.model(batch))


class UniCalRefineStage:
    """
    Wraps a trained UniCal refiner (UniCal-M or UniCal-S) as a cascade stage.

    Args:
        model:        Trained ``unical.models.module.UniCal`` module (eval).
        preprocessor: UniCal ``DataPreprocessor`` that projects the LiDAR into the
                      image plane with the current extrinsic estimate.
    """

    def __init__(self, model: UniCal, preprocessor: DataPreprocessor) -> None:
        self.model = model.eval()
        self.preprocessor = preprocessor

    @torch.no_grad()
    def predict(self, raw: RawSample, T_current: Transform, device: torch.device) -> Transform:
        img_pp, lidar_map = self.preprocessor(
            raw.image_rgb.copy(), raw.pcl.copy(), T_current.matrix, raw.K.copy()
        )
        batch = UniCalBatch(
            img=torch.from_numpy(img_pp).permute(2, 0, 1)[None].to(device),
            lidar_map=torch.from_numpy(lidar_map).permute(2, 0, 1)[None].to(device),
            target_reg=(torch.zeros(1, 3, device=device), torch.eye(3, device=device)[None]),
            pcl=torch.from_numpy(raw.pcl)[None].to(device),
            metadata=[{"T_init": T_current, "T_gt": raw.T_gt, "K": raw.K}],
        )
        return _pred_to_transform(self.model(batch))


class CascadeCalibrator:
    """
    Run an ordered list of stages to recover the extrinsic from ``T_init``.

    Args:
        stages: Ordered cascade stages (coarse → fine).
        device: Device to run inference on.
    """

    def __init__(
        self,
        stages: list[CascadeStage],
        device: torch.device | str = "cpu",
    ) -> None:
        assert stages, "CascadeCalibrator needs at least one stage"
        self.stages = stages
        self.device = torch.device(device)

    def calibrate(self, raw: RawSample) -> Transform:
        """Return the final extrinsic estimate after running every stage."""
        T = raw.T_init
        for stage in self.stages:
            residual = stage.predict(raw, T, self.device)
            T = residual.inverse() @ T  # apply the recalibration
        return T

    def evaluate(self, raws: list[RawSample]) -> dict[str, float]:
        """Run the cascade over ``raws`` and return UniCal-style calibration metrics."""
        metrics = CalibMetrics()
        for raw in raws:
            T_pred = self.calibrate(raw)
            metrics.add(T_pred, raw.T_gt)  # error = T_pred vs ground truth
        return metrics.all_metrics()
