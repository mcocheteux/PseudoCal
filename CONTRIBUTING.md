# Contributing to PseudoCal

Thanks for your interest in improving PseudoCal! Contributions of all sizes are welcome —
bug fixes, tests, docs, or new depth backends.

## Development setup

```bash
uv sync --extra dev --extra logger
```

## Before opening a PR

The same three checks run in [CI](.github/workflows/ci.yml); run them locally first.

1. **Lint & format** — the code must be clean under ruff:
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   ```
2. **Test** — the suite must pass:
   ```bash
   uv run pytest -v
   ```
   Tests stay offline (the synthetic `DummyDepthEstimator` is used; no model weights are
   downloaded). Tests that need the `unical-plus` dependency `importorskip` cleanly.
3. **Smoke train** — if you touched the model or data path, run a one-step check:
   ```bash
   python train.py data_dir=/path/to/kitti_raw experiment=debug depth=dummy \
     trainer.fast_dev_run=true
   ```

## Style

- Match the surrounding code (it mirrors the
  [unical-plus](https://github.com/mcocheteux/unical-plus) style): `from __future__ import
  annotations`, typed signatures, module/class docstrings, and comments that explain *why*.
- Geometry runs in float32 even under AMP — keep rigid-transform math out of autocast.
- New behaviour should come with a test in `tests/`.

## License

By contributing you agree your contributions are licensed under **CC BY-NC 4.0**.
