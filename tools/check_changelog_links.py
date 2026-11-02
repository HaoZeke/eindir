"""Check that CHANGELOG.md release headings link to v-prefixed tags.

Every ``## [X.Y.Z](URL)`` heading must link to ``<repository>/tree/vX.Y.Z``,
with the repository URL taken from Cargo.toml. With ``--remote NAME`` the
script also lists the tags that are missing from that git remote.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

from cargo_version import REPOSITORY

CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"
HEADING = re.compile(r"^## \[(?P<version>[^\]]+)\]\((?P<url>[^)]*)\)", re.MULTILINE)
TAG_PREFIX = "v"


def remote_tags(remote: str) -> set[str]:
    out = subprocess.run(
        ["git", "ls-remote", "--tags", "--refs", remote],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {line.split("refs/tags/", 1)[1] for line in out.splitlines()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", help="git remote whose tags must exist")
    args = parser.parse_args()

    headings = HEADING.findall(CHANGELOG.read_text(encoding="utf-8"))
    if not headings:
        print("no release headings found in CHANGELOG.md", file=sys.stderr)
        return 1

    bad = []
    for version, url in headings:
        expected = f"{REPOSITORY}/tree/{TAG_PREFIX}{version}"
        if url != expected:
            bad.append(f"{version}: {url} (expected {expected})")
    for line in bad:
        print(f"bad changelog link {line}", file=sys.stderr)

    missing = []
    if args.remote:
        tags = remote_tags(args.remote)
        for version, _ in headings:
            tag = f"{TAG_PREFIX}{version}"
            status = "ok" if tag in tags else "MISSING"
            print(f"{tag}: {status}")
            if tag not in tags:
                missing.append(tag)

    if bad or missing:
        return 1
    print(f"{len(headings)} changelog links use the {TAG_PREFIX}-prefixed tag pattern")
    return 0


if __name__ == "__main__":
    sys.exit(main())
