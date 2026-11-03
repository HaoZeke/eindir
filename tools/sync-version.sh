#!/usr/bin/env bash
# Regenerate every version copy listed in tools/version-files.txt from
# Cargo.toml. Uses cogapp through `python -m cogapp` because `cog` on
# PATH may be cocogitto.
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON:-python3}" -m cogapp -r -I tools @tools/version-files.txt
