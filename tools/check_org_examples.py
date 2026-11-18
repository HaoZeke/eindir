#!/usr/bin/env python3
"""Run the Python blocks of the manual and check its #+INCLUDE targets.

    python tools/check_org_examples.py [FILE.org ...]

Each ``#+begin_src python`` block in docs/orgmode runs in a fresh
interpreter (the one running this script), with the repository root as the
working directory, so eindir must be importable. A block whose header has
``:eval no`` is skipped. Every ``#+INCLUDE:`` must name an existing file.
Reports every failure and exits 1 if there was any.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORG = ROOT / "docs" / "orgmode"

BEGIN = re.compile(r"^\s*#\+begin_src\s+python\b(.*)$", re.I)
END = re.compile(r"^\s*#\+end_src\b", re.I)
INCLUDE = re.compile(r"^\s*#\+include:\s*(.*)$", re.I)


def blocks(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        m = BEGIN.match(lines[i])
        if not m:
            i += 1
            continue
        start = i + 1
        j = start
        while j < len(lines) and not END.match(lines[j]):
            j += 1
        yield start, m.group(1), "\n".join(lines[start:j])
        i = j + 1


def includes(path: Path):
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        m = INCLUDE.match(line)
        if m:
            yield number, shlex.split(m.group(1))[0]


def main(argv: list[str]) -> int:
    files = [Path(a).resolve() for a in argv] or sorted(ORG.rglob("*.org"))
    failures = 0
    ran = 0
    for path in files:
        rel = path.relative_to(ROOT)
        for number, target in includes(path):
            if not (path.parent / target).is_file():
                print(f"{rel}:{number}: #+INCLUDE target {target} does not exist")
                failures += 1
        for line, header, code in blocks(path):
            if re.search(r":eval\s+no\b", header):
                continue
            ran += 1
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                failures += 1
                print(f"{rel}:{line}: Python block failed")
                print(result.stderr.rstrip())
    print(f"{ran} Python block(s) run, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
