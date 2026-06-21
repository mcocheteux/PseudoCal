"""Hydra config composition / instantiation smoke tests."""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

pytest.importorskip("unical")

import os  # noqa: E402

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))


def _compose(overrides: list[str], config_name: str = "train"):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        return compose(config_name=config_name, overrides=overrides)


def test_pseudopillars_model_instantiates() -> None:
    # depth=dummy keeps it offline (no HF download).
    cfg = _compose(
        ["data_dir=/tmp/kitti", "model=pseudopillars", "data=kitti_pillars", "depth=dummy"]
    )
    model = instantiate(cfg.model)
    assert model.__class__.__name__ == "PseudoPillars"
    # Pillar channels feed the backbone stem (2 modalities concatenated).
    assert model.pillars_pseudo.out_channels == cfg.model.pillar_channels


@pytest.mark.parametrize("stage", ["unical_m", "unical_s"])
def test_unical_refiner_models_instantiate(stage: str) -> None:
    cfg = _compose(["data_dir=/tmp/kitti", f"model={stage}", "data=kitti_m"])
    model = instantiate(cfg.model)
    assert model.__class__.__name__ == "UniCal"


def test_data_configs_instantiate() -> None:
    for data in ["kitti_pillars", "kitti_m", "kitti_s"]:
        cfg = _compose(["data_dir=/tmp/kitti", f"data={data}", "depth=dummy"])
        dm = instantiate(cfg.data)
        assert dm is not None


def test_cascade_config_composes() -> None:
    cfg = _compose(["data_dir=/tmp/kitti", "depth=dummy"], config_name="cascade")
    # Each stage's model config is present and the dataset/preprocessor instantiate.
    assert "pseudopillars" in cfg and "unical_m" in cfg and "unical_s" in cfg
    instantiate(cfg.preprocessor)
