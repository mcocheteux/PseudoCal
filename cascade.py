"""
End-to-end PseudoCal cascade evaluation.

Loads the three trained stages (PseudoPillars → UniCal-M → UniCal-S), runs them in
sequence on the test split starting from a large decalibration, and reports the
final calibration error (the same MAE/STD metrics UniCal uses).

Usage:
    python cascade.py data_dir=/path/to/kitti_raw \
        +ckpt.pseudopillars=logs/checkpoints/pseudopillars/last.ckpt \
        +ckpt.unical_m=logs/checkpoints/unical_m/last.ckpt \
        +ckpt.unical_s=logs/checkpoints/unical_s/last.ckpt \
        limit=200
"""

from __future__ import annotations

import hydra
import pytorch_lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from pseudocal.cascade.runner import (
    CascadeCalibrator,
    PseudoPillarsStage,
    RawSample,
    UniCalRefineStage,
)

torch.set_float32_matmul_precision("high")


def _resolve_device(accelerator: str) -> torch.device:
    if accelerator == "cuda" or (accelerator == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def _load_weights(
    model: L.LightningModule, ckpt: str | None, device: torch.device
) -> L.LightningModule:
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")["state_dict"]
        # strict=False: PseudoPillars omits the frozen depth weights from its
        # checkpoint (they are reloaded from the pretrained source on construction).
        missing, unexpected = model.load_state_dict(state, strict=False)
        leaked = [k for k in unexpected if not k.startswith("depth.")]
        if leaked:
            raise RuntimeError(f"Unexpected checkpoint keys for {type(model).__name__}: {leaked}")
    return model.to(device).eval()


@hydra.main(config_path="configs", config_name="cascade", version_base="1.3")
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed, workers=True)
    device = _resolve_device(cfg.accelerator)

    # ── Stages ────────────────────────────────────────────────────────
    pseudo_model = _load_weights(instantiate(cfg.pseudopillars), cfg.ckpt.pseudopillars, device)
    unical_m = _load_weights(instantiate(cfg.unical_m), cfg.ckpt.unical_m, device)
    unical_s = _load_weights(instantiate(cfg.unical_s), cfg.ckpt.unical_s, device)
    preprocessor = instantiate(cfg.preprocessor)

    stages = [
        PseudoPillarsStage(
            pseudo_model,
            width=cfg.dataset.width,
            height=cfg.dataset.height,
            edge_low=cfg.dataset.edge_low,
            edge_high=cfg.dataset.edge_high,
            edge_dilate=cfg.dataset.edge_dilate,
        ),
        UniCalRefineStage(unical_m, preprocessor),
        UniCalRefineStage(unical_s, preprocessor),
    ]
    calibrator = CascadeCalibrator(stages, device=device)

    # ── Raw test frames ───────────────────────────────────────────────
    dataset = instantiate(cfg.dataset)
    n = len(dataset) if cfg.limit is None else min(int(cfg.limit), len(dataset))
    raws = [RawSample(**dataset.raw_sample(i)) for i in range(n)]
    print(f"[cascade] evaluating {n} frames on {device} …")

    metrics = calibrator.evaluate(raws)

    print("\n=== PseudoCal cascade — final calibration error ===")
    for key in ("rot/global/MAE", "rot/global/STD", "trans/global/MAE", "trans/global/STD"):
        if key in metrics:
            unit = "deg" if key.startswith("rot") else "cm"
            print(f"  {key:22s}: {metrics[key]:.4f} {unit}")
    print()
    for k in sorted(metrics):
        print(f"  {k:22s}: {metrics[k]:.4f}")


if __name__ == "__main__":
    main()
