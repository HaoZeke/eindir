#!/usr/bin/env bash
# Fail when any copy of the version disagrees with Cargo.toml.
set -euo pipefail
cd "$(dirname "$0")/.."

"${PYTHON:-python3}" -m cogapp --check --diff \
    --check-fail-msg='run tools/sync-version.sh' \
    -I tools @tools/version-files.txt

if ! cargo metadata --locked --format-version 1 >/dev/null; then
    echo "Cargo.lock is out of date; run cargo update --workspace" >&2
    exit 1
fi

"${PYTHON:-python3}" tools/check_changelog_links.py

echo "All version copies match Cargo.toml"
