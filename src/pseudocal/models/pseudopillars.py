"""
PseudoPillars: the coarse, initialisation-free first stage of PseudoCal.

Pipeline (see the paper, arXiv:2309.09855):

    camera image ──► monocular metric depth ──► back-project ──► pseudo-LiDAR ┐
                                                                              ├─► two
    real LiDAR ──► transform by current extrinsic estimate ─────────────────►┘  BEV
                                                                                 pillar
                                                                                 images
                                            │
                                            ▼
                          MobileViT fusion ──► split regression head ──► T_decal

Because both clouds are encoded in **3-D** (full bird's-eye-view, not the camera
field of view), the network can recover *large* miscalibrations — up to ±180° yaw
— with **no initial estimate**, which is the whole point of PseudoCal's first
stage. The two downstream UniCal refiners then polish the result.

This module deliberately mirrors :class:`unical.models.module.UniCal` (same
shared-step / logging / optimiser structure) and reuses UniCal's losses and
metrics unchanged — only the ``forward`` (pseudo-LiDAR + pillars) is new.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytorch_lightning as L
import torch
import torch.nn as nn
from unical.losses.combined import CombinedLoss
from unical.models.backbone import MobileViTBackbone
from unical.models.head import SplitRegressionHead
from unical.utils.metrics import CalibMetrics
from unical.utils.transform import Transform, rotation_6d_to_matrix

from pseudocal.data.dataset import PseudoBatch
from pseudocal.depth.base import DepthEstimator
from pseudocal.pillars import PillarFeatureNet, PillarGrid
from pseudocal.pseudolidar import backproject
from pseudocal.scale import estimate_range_scale, point_range_quantiles


class PseudoPillars(L.LightningModule):
    """
    PseudoPillars LightningModule.

    Args:
        depth:        Frozen monocular metric-depth estimator.
        backbone:     MobileViT fusion backbone (``pretrained=None``,
                      ``img_channels == lidar_channels == pillar_channels``).
        head:         Split translation/rotation regression head.
        loss:         CombinedLoss (regression + spatial), reused from UniCal.
        grid:         BEV pillar grid (camera-frame x/z plane).
        pillar_channels:    Channels per pillar pseudo-image (per modality).
        backproject_stride: Pixel sub-sampling stride for pseudo-LiDAR.
        min_depth/max_depth: Depth range (m) kept when back-projecting.
        max_pseudo_points:  Padded length of the pseudo-LiDAR cloud.
        scale_align:        Depth-scale correction: ``none`` | ``range`` (rotation-invariant
                            range matching to the real LiDAR) | ``range+residual`` (+ a small
                            learned correction). Default ``range+residual``.
        scale_bias:         Fixed multiplicative scale error injected on the pseudo-LiDAR
                            (1.0 = none; used for the eval sensitivity sweep).
        scale_jitter:       Train-time per-sample random scale augmentation (±fraction).
        lr / weight_decay / warmup_epochs: optimiser schedule (as UniCal).
    """

    def __init__(
        self,
        depth: DepthEstimator,
        backbone: MobileViTBackbone,
        head: SplitRegressionHead,
        loss: CombinedLoss,
        grid: PillarGrid,
        pillar_channels: int = 32,
        backproject_stride: int = 2,
        min_depth: float = 1.0,
        max_depth: float = 80.0,
        max_pseudo_points: int = 30000,
        scale_align: str = "range+residual",
        scale_jitter: float = 0.0,
        scale_bias: float = 1.0,
        lr: float = 3e-5,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["depth", "backbone", "head", "loss", "grid"])

        self.depth = depth
        self.pillars_pseudo = PillarFeatureNet(grid, pillar_channels)
        self.pillars_real = PillarFeatureNet(grid, pillar_channels)
        self.backbone = backbone
        self.head = head
        self.loss_fn = loss

        self.grid = grid

        # Depth-scale alignment. `scale_bias`/`scale_jitter` *inject* a metric-depth
        # scale error on the pseudo-LiDAR (eval sweep / train augmentation); `scale_align`
        # *corrects* it by anchoring to the real LiDAR's range distribution.
        if scale_align not in ("none", "range", "range+residual"):
            raise ValueError(f"scale_align must be none|range|range+residual, got {scale_align!r}")
        self._scale_quantiles = (0.5, 0.7, 0.9)
        self.scale_residual: nn.Module | None = None
        if scale_align == "range+residual":
            q = len(self._scale_quantiles)
            self.scale_residual = nn.Sequential(nn.Linear(2 * q, 16), nn.ReLU(), nn.Linear(16, 1))

        self._val_metrics = CalibMetrics()
        self._test_metrics = CalibMetrics()

    # ------------------------------------------------------------------
    # Geometry helpers (run in float32, outside AMP autocast)
    # ------------------------------------------------------------------

    def _pseudo_cloud(self, batch: PseudoBatch) -> torch.Tensor:
        """
        Camera-frame pseudo-LiDAR ``(B, P, 4)`` (``[x, y, z, valid]``).

        Uses pre-computed points when the dataset cached them (the common training
        path — the frozen depth network is run once offline, not every step); falls
        back to a live monocular-depth forward otherwise (e.g. cascade inference).
        """
        if batch.pseudo_pcl is not None:
            pts = batch.pseudo_pcl.float()  # (B, P, 3), zero-padded
            valid = pts.abs().sum(-1) > 0
            pts = pts * valid.unsqueeze(-1)
            cloud = torch.cat([pts, valid.float().unsqueeze(-1)], dim=-1)  # (B, P, 4)
        else:
            depth = self.depth.estimate(batch.image).float()  # (B, 1, H, W)
            cloud = backproject(
                depth,
                batch.K.float(),
                batch.edge_mask,
                stride=self.hparams.backproject_stride,
                min_depth=self.hparams.min_depth,
                max_depth=self.hparams.max_depth,
                max_points=self.hparams.max_pseudo_points,
            )
        return self._perturb_scale(cloud)

    def _perturb_scale(self, cloud: torch.Tensor) -> torch.Tensor:
        """
        Inject a metric-depth scale error on the pseudo-LiDAR: a fixed ``scale_bias`` (eval
        sensitivity sweep) and, during training, an extra per-sample random ``scale_jitter``.
        Both default to a no-op. This *simulates* depth-scale drift; ``_align_scale`` corrects it.
        """
        bias = float(self.hparams.scale_bias)
        jitter = float(self.hparams.scale_jitter)
        if bias == 1.0 and not (self.training and jitter > 0.0):
            return cloud
        b = cloud.shape[0]
        factor = torch.full((b, 1, 1), bias, device=cloud.device, dtype=cloud.dtype)
        if self.training and jitter > 0.0:
            factor = factor * (1.0 + (torch.rand(b, 1, 1, device=cloud.device) * 2 - 1) * jitter)
        return torch.cat([cloud[..., :3] * factor, cloud[..., 3:]], dim=-1)

    def _align_scale(self, pseudo: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """
        Rescale the pseudo-LiDAR so its range distribution matches the real LiDAR's
        (rotation-invariant; see :mod:`pseudocal.scale`). The closed-form factor is detached
        for stability; ``range+residual`` adds a small learned multiplicative correction.
        """
        q = self._scale_quantiles
        s = estimate_range_scale(pseudo, real, quantiles=q).detach()  # (B,)
        if self.scale_residual is not None:
            qp = point_range_quantiles(pseudo, q).clamp_min(1e-6).log()
            qr = point_range_quantiles(real, q).clamp_min(1e-6).log()
            feat = torch.nan_to_num(torch.cat([qp, qr], dim=-1))  # (B, 2Q)
            s = s * torch.exp(self.scale_residual(feat).squeeze(-1))
        return torch.cat([pseudo[..., :3] * s[:, None, None], pseudo[..., 3:]], dim=-1)

    @staticmethod
    def _real_cloud(batch: PseudoBatch) -> torch.Tensor:
        """Real LiDAR mapped to the camera frame with the (decalibrated) extrinsic."""
        device = batch.pcl.device
        pcl = batch.pcl.float()  # (B, N, 4)
        xyz = pcl[..., :3]
        valid = pcl[..., :3].abs().sum(-1) > 0  # drop all-zero padding

        T = torch.stack([m["T_init"].to_torch(device) for m in batch.metadata])  # (B,4,4)
        R, t = T[:, :3, :3], T[:, :3, 3]
        xyz_cam = torch.einsum("bij,bnj->bni", R, xyz) + t[:, None, :]
        xyz_cam = xyz_cam * valid.unsqueeze(-1)  # zero invalid rows
        return torch.cat([xyz_cam, valid.float().unsqueeze(-1)], dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: PseudoBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (trans_pred (B,3), rot6d_pred (B,6))."""
        # Build both clouds in full precision (geometry is fp32-sensitive); the
        # learnable pillar/backbone path below still benefits from AMP.
        with torch.autocast(device_type=self.device.type, enabled=False):
            pseudo = self._pseudo_cloud(batch)
            real = self._real_cloud(batch)
            if self.hparams.scale_align != "none":
                pseudo = self._align_scale(pseudo, real)

        img_pseudo = self.pillars_pseudo(pseudo)  # (B, C, H, W)
        img_real = self.pillars_real(real)  # (B, C, H, W)
        features = self.backbone(SimpleNamespace(img=img_pseudo, lidar_map=img_real))
        return self.head(features)

    # ------------------------------------------------------------------
    # Shared step (identical bookkeeping to UniCal)
    # ------------------------------------------------------------------

    def _step(
        self, batch: PseudoBatch
    ) -> tuple[dict[str, torch.Tensor], list[Transform], list[Transform]]:
        pred = self(batch)
        losses = self.loss_fn(pred, batch)

        B = pred[0].shape[0]
        pred_t = pred[0].detach().float().cpu().numpy()
        pred_R = rotation_6d_to_matrix(pred[1]).detach().float().cpu().numpy()
        tgt_t = batch.target_reg[0].detach().float().cpu().numpy()
        tgt_R = batch.target_reg[1].detach().float().cpu().numpy()
        pred_Ts = [Transform.from_rotation_translation(pred_R[i], pred_t[i]) for i in range(B)]
        target_Ts = [Transform.from_rotation_translation(tgt_R[i], tgt_t[i]) for i in range(B)]
        return losses, pred_Ts, target_Ts

    # ------------------------------------------------------------------
    # Training / validation / test
    # ------------------------------------------------------------------

    def training_step(self, batch: PseudoBatch, batch_idx: int) -> torch.Tensor:
        losses, _, _ = self._step(batch)
        B = batch.pcl.shape[0]
        self.log_dict(
            {f"train/{k}": v for k, v in losses.items()},
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
            batch_size=B,
        )
        return losses["loss"]

    def validation_step(self, batch: PseudoBatch, batch_idx: int) -> None:
        losses, pred_Ts, target_Ts = self._step(batch)
        B = batch.pcl.shape[0]
        self.log_dict(
            {f"val/{k}": v for k, v in losses.items()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )
        for p, t in zip(pred_Ts, target_Ts):
            self._val_metrics.add(p, t)

    def on_validation_epoch_end(self) -> None:
        metrics = self._val_metrics.all_metrics()
        self.log_dict({f"val/{k}": v for k, v in metrics.items()}, sync_dist=True)
        self._val_metrics.clear()

    def test_step(self, batch: PseudoBatch, batch_idx: int) -> None:
        losses, pred_Ts, target_Ts = self._step(batch)
        B = batch.pcl.shape[0]
        self.log_dict(
            {f"test/{k}": v for k, v in losses.items()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )
        for p, t in zip(pred_Ts, target_Ts):
            self._test_metrics.add(p, t)

    def on_test_epoch_end(self) -> None:
        metrics = self._test_metrics.all_metrics()
        self.log_dict({f"test/{k}": v for k, v in metrics.items()}, sync_dist=True)
        self._test_metrics.clear()

    # ------------------------------------------------------------------
    # Checkpoint I/O — exclude the frozen depth estimator
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # The depth estimator is frozen and re-loaded from its pretrained source,
        # so its (large) weights do not belong in our checkpoints.
        sd = checkpoint["state_dict"]
        for k in [k for k in sd if k.startswith("depth.")]:
            del sd[k]

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # Re-insert the freshly constructed depth weights so strict loading succeeds.
        sd = checkpoint["state_dict"]
        for k, v in self.state_dict().items():
            if k.startswith("depth.") and k not in sd:
                sd[k] = v

    # ------------------------------------------------------------------
    # Optimiser (AdamW + cosine with optional warmup — matches UniCal)
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:
        # The frozen depth estimator exposes no trainable parameters, but filter
        # explicitly so a non-frozen estimator could never leak into the optimiser.
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        max_epochs = self.trainer.max_epochs
        warmup = min(self.hparams.warmup_epochs, max(max_epochs - 1, 0))
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(max_epochs - warmup, 1), eta_min=1e-7
        )
        if warmup > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=1e-2, total_iters=warmup
            )
            scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warmup_sched, cosine], milestones=[warmup]
            )
        else:
            scheduler = cosine
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
