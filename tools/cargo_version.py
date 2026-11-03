"""Crate metadata from Cargo.toml, the single source of the release version.

cogapp markers import this module (run cogapp with ``-I tools``) to
regenerate every copy of the version that a tool cannot read from
Cargo.toml itself.
"""

import tomllib
from pathlib import Path

CARGO_TOML = Path(__file__).resolve().parent.parent / "Cargo.toml"

with CARGO_TOML.open("rb") as _fh:
    PACKAGE = tomllib.load(_fh)["package"]

VERSION = PACKAGE["version"]
REPOSITORY = PACKAGE["repository"]
