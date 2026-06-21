"""
Depth Anything V2 (metric) estimator — the default pseudo-LiDAR depth source.

We use the *metric, outdoor* checkpoints, which are fine-tuned to predict depth in
metres on driving-style scenes (Virtual KITTI), so the back-projected pseudo-LiDAR
shares the real LiDAR's absolute scale. This is a strict upgrade over the paper's
original GLPN estimator (newer DINOv2 backbone, stronger zero-shot generalisation).

Reference: Yang et al., "Depth Anything V2", NeurIPS 2024.
HF model card: depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pseudocal.depth.base import DepthEstimator

# ImageNet statistics — the convention the rest of the PseudoCal pipeline uses to
# normalise images (see unical.utils.geometry.imagenet_normalize). We undo this
# before re-normalising with the model's own statistics.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DepthAnythingV2MetricEstimator(DepthEstimator):
    """
    Frozen Depth Anything V2 metric-depth estimator (HuggingFace ``transformers``).

    Args:
        model_id:    HF hub id of a metric Depth-Anything-V2 checkpoint.
        infer_size:  Square side (multiple of 14) the image is resized to before
                     the ViT. Depth is bilinearly resampled back to the input size.
        max_depth:   Upper clamp on the returned metric depth, in metres. Acts as a
                     guard against rare large outliers from the network.
    """

    def __init__(
        self,
        model_id: str = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        infer_size: int = 518,
        max_depth: float = 80.0,
    ) -> None:
        super().__init__()
        # Imported lazily so the (heavy) transformers import only happens when this
        # estimator is actually instantiated — tests use the Dummy estimator.
        from transformers import AutoImageProcessor, DepthAnythingForDepthEstimation

        self.model = DepthAnythingForDepthEstimation.from_pretrained(model_id)

        # Pull the model's own normalisation statistics from its image processor so
        # we feed it pixel values in exactly the distribution it was trained on.
        processor = AutoImageProcessor.from_pretrained(model_id)
        mean = getattr(processor, "image_mean", _IMAGENET_MEAN)
        std = getattr(processor, "image_std", _IMAGENET_STD)
        self.register_buffer("_in_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_in_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))
        self.register_buffer("_model_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("_model_std", torch.tensor(std).view(1, 3, 1, 1))

        self.infer_size = infer_size
        self.max_depth = max_depth
        self.freeze()

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        b, _, h, w = images.shape

        # Re-normalise: undo the pipeline's ImageNet normalisation, then apply the
        # model's expected statistics (identical for DA-V2, but kept explicit so the
        # estimator is correct for any checkpoint).
        pixel = images * self._in_std + self._in_mean  # -> [0, 1] RGB
        pixel = (pixel - self._model_mean) / self._model_std

        # The ViT needs a side that is a multiple of the patch size (14).
        pixel = F.interpolate(
            pixel,
            size=(self.infer_size, self.infer_size),
            mode="bilinear",
            align_corners=False,
        )

        depth = self.model(pixel_values=pixel).predicted_depth  # (B, h', w'), metres
        depth = depth.view(b, 1, depth.shape[-2], depth.shape[-1])
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=False)
        return depth.clamp(min=0.0, max=self.max_depth)
