"""
GLPN monocular depth estimator — the paper-faithful pseudo-LiDAR depth source.

The original PseudoCal paper (arXiv:2309.09855) builds pseudo-LiDAR from a GLPN
depth map. This estimator reproduces that choice; the default pipeline uses the
newer :class:`~pseudocal.depth.depth_anything.DepthAnythingV2MetricEstimator`
instead, but GLPN is kept for faithful comparison / ablation.

Reference: Kim et al., "Global-Local Path Networks for Monocular Depth
Estimation with Vertical CutDepth", 2022.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pseudocal.depth.base import DepthEstimator

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class GLPNDepthEstimator(DepthEstimator):
    """
    Frozen GLPN depth estimator (HuggingFace ``transformers``).

    Args:
        model_id:   HF hub id of a GLPN checkpoint (KITTI by default).
        infer_size: Square side the image is resized to before the encoder; GLPN
                    is fully convolutional, so any multiple of 32 works.
        max_depth:  Upper clamp on the returned depth, in metres.
    """

    def __init__(
        self,
        model_id: str = "vinvino02/glpn-kitti",
        infer_size: int = 480,
        max_depth: float = 80.0,
    ) -> None:
        super().__init__()
        from transformers import GLPNForDepthEstimation, GLPNImageProcessor

        self.model = GLPNForDepthEstimation.from_pretrained(model_id)

        processor = GLPNImageProcessor.from_pretrained(model_id)
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

        pixel = images * self._in_std + self._in_mean
        pixel = (pixel - self._model_mean) / self._model_std
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
