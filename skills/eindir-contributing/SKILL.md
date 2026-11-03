---
name: eindir-contributing
description: Prepare a change to eindir for review. Covers which files to edit for code, the C ABI, the manual and the version, the commit types cog accepts, towncrier fragments, the Org to Sphinx docs build, regenerating llms.txt, bumping the version from Cargo.toml, and the CI failures on main that predate any change. Use before committing to eindir, when the lint, version-drift or skills CI jobs fail, or when preparing a release.
license: See LICENSE in the eindir repository
compatibility: The preflight script needs Python 3.11 or newer and git. Its version check also needs cogapp importable from the same Python and cargo 1.88 or newer, and is skipped otherwise. The docs build needs pixi, and network access the first time, when Emacs fetches the ox-rst package.
---

# Contributing to eindir

Before you commit, run `scripts/preflight.py --base origin/main` from this skill.
It checks the fragments, `llms.txt`, the version copies and your commit subjects, as described below.

## Where a change goes

- Rust API: edit `src/`, along with its tests in `tests/` or the module's `#[cfg(test)]` and the Org page that documents it.
- Python extension: edit `src/python.rs` and `src/py_objective.rs` (`eindir._core`), then update `pytest/` and `skills/eindir-python/scripts/surface.py`.
- Pure-Python package: edit `python/eindir/` and its tests in `pytest/`.
- C ABI: edit `src/ffi.rs`, from which cbindgen writes `eindir-core.h`.
  `tests/ffi_smoke.c`, `tests/header_check.c`, `tests/header_check.cpp` and `include/eindir-core.hpp` must change with it.
- Manual: edit `docs/orgmode/**/*.org`, and regenerate `llms.txt` and `llms-full.txt`.
- Version: edit `version` in `Cargo.toml` only, as "Version" below explains.

`docs/source/*.rst` and `docs/build/` are generated and ignored by git, so edit the Org files.

## Commit subjects

`type(scope): description`, or `type: description`, with no period at the end.
cog accepts its built-in types (`feat`, `fix`, `docs`, `test`, `ci`, `chore`, `style`, `refactor`, `perf`, `build`, `revert`) and the ones `cog.toml` adds (`enh`, `bug`, `maint`, `prj`, `doc`, `tst`, `bld`, `gen`, `dat`, `bench`).
It ignores case, which is why the `MAINT:` and `DOC:` commits of 2023 and early 2024 pass, but every commit since then writes the type in lowercase.
Do the same.
`feat`, `fix`, `docs`, `chore`, `ci` and `test` are the common ones.

## News fragments

Every change outside `docs/` gets a fragment in `docs/newsfragments/`, named `+<slug>.<type>.rst`, or `123.<type>.rst` to link issue or pull request 123.
`<type>` is one of `added`, `changed`, `fixed`, `removed`, `deprecated`, `security`, `dev` and `misc`.
Write a sentence or two in plain prose, with code in double backticks as in reStructuredText.
The fragment for the autodiff feature, an `added` one, read:

```rst
Optional ``autodiff`` feature: ``ForwardAd`` wraps a ``DualField`` written
once over ``num-dual`` and supplies an exact fused gradient.
```

towncrier also reads `.md` fragments (the 0.7.0 one was `c-header.fixed.md`, with single backticks) and copies the text into `CHANGELOG.md` as written.
Without a warning, it skips a file whose name has no known type or that sits in a subdirectory, and it drops an empty one.
A name that neither starts with `+` nor consists of digits becomes a link to an issue of that name.
`scripts/preflight.py` catches all four.

The towncrier draft, `pixi run towncrier-draft`, previews the next release notes under the version in `towncrier.toml`.
At commit 8613085 that is 0.7.0, which `CHANGELOG.md` already has, so bump the version before building the notes for real.
Fragments are consumed only when a release is prepared.

## Docs

```bash
pixi install -e docs
pixi run -e docs docbld     # Org to RST, then Sphinx HTML in docs/build/
```

`docbld` runs `orgbld` (Emacs exports `docs/orgmode/` to `docs/source/`) and installs `sphinx-rustdocgen` into the environment on its first run.
At commit 8613085 it built with no Sphinx warnings.
Keep it that way.

After any change to the Org files, `AGENTS.md` or a skill, regenerate `llms.txt` and `llms-full.txt`.

```bash
python3 scripts/llms/build_llms.py
```

## Version

The `version` in `Cargo.toml` is the only copy written by hand.
`pyproject.toml` reads it through maturin, the C and Python libraries report it at run time, and `meson.build`, `CMakeLists.txt`, `pixi.toml` and `towncrier.toml` hold generated copies between cogapp markers.
To bump it:

```bash
python3 -m pip install cogapp==3.6.0   # the version CI pins
tools/sync-version.sh                  # rewrite the generated copies
cargo update --workspace               # Cargo.lock
tools/check-version-drift.sh
```

- Call cogapp as `python3 -m cogapp`, as the scripts do: `cog` on PATH may be cocogitto.
- Do not edit between `[[[cog` and `[[[end]]]` by hand.
- After `sync-version.sh` alone, the drift check still fails with `Cargo.lock is out of date; run cargo update --workspace`.
- With cargo older than 1.88 it prints that same message although the lock file is fine.
  The real error, a few lines above, is "feature `edition2024` is required".
  Use `RUSTUP_TOOLCHAIN=stable`.

## CI on main

At commit 8613085, 16 of the 17 GitHub checks passed.
The Lint job fails at its first step, `cargo fmt --check`, with 11 hunks in 6 files under `src/`.
Its later steps do not run.
Run by hand, `cargo clippy --all-features --all-targets -- -D warnings` reports 4 errors (three unsafe functions without a `# Safety` section and one hand-built nul-terminated string), and `cog check`, which reads the whole history, rejects the two `REL:` commits of 2024-02-17.
Lint is red before your change, so keep unrelated reformatting out of your commit.

## Scripts

`scripts/preflight.py` prints one line per check: ok, FAIL or skip.

```bash
python3 scripts/preflight.py                        # fragments, llms.txt, version copies
python3 scripts/preflight.py --base origin/main     # also subjects and a fragment for the change
python3 scripts/preflight.py --subject "fix(ffi): reject NULL bounds"
```

It exits 1 when a check fails.
The version check needs cogapp in the Python that runs the script and cargo 1.88 or newer.
Under rustup, set `RUSTUP_TOOLCHAIN=stable`.
