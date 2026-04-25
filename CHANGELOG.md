# Changelog

## [0.2.0] - 2026

### Changed

- BREAKING: build system migrated from PDM (`pdm-backend`) to `maturin`
  mixed mode. Python sources now live under `python/eindir/`; the wheel
  ships a Rust extension as `eindir._core` alongside the existing
  pure-Python helpers.
- BREAKING: Python tests moved from `tests/` to `pytest/`.

### Added

- Rust crate `eindir-core` (lib + cdylib + staticlib) scaffolding for
  the typed component algebra landing in v0.3.0.
- C/C++ bindings via `cargo-c` (`pkg-config`, headers, hand-written
  C++ companion).
- Build-system glue (`meson.build`, `meson_options.txt`, `CMakeLists.txt`)
  for downstream native consumers.
- Pixi workspace with `default`, `python`, and (in the next minor)
  `docs` environments.
- Orgmode + Sphinx (shibuya theme) documentation pipeline replacing
  the legacy MyST + furo setup.

### Removed

- PDM tooling: `pdm.lock`, PDM-specific sections in `pyproject.toml`.
- `environment.yml` (replaced by `pixi.toml`).
- `tbump.toml` (replaced by `cog bump`).
- `towncrier.toml` and the `changelog.d/` news-fragment system
  (replaced by `cog changelog` driven directly from Conventional Commits).
- `.pre-commit-config.yaml` (CI handles fmt and clippy gating; revisit
  if pre-commit becomes a hard requirement).

## [0.1.0](https://github.com/HaoZeke/eindir/tree/0.1.0) - 17-02-2024


No significant changes.


## [0.0.5](https://github.com/HaoZeke/eindir/tree/0.0.5) - 17-02-2024


No significant changes.
