#!/usr/bin/env python3
"""Check the links and DOIs of the manual and the top-level documents.

    python tools/check_links.py [FILE ...]

Scans README.md, CHANGELOG.md, AGENTS.md, docs/orgmode, docs/newsfragments
and skills/*/SKILL.md by default. Every DOI must be registered: the
doi.org handle API must answer with responseCode 1. Every other http(s)
URL must answer a GET with a status below 400. Hosts in CHALLENGE answer
scripted clients with 403 and a challenge page, so 403 counts as reachable
for them. Release tag links (TAG_LINK) are left to
``tools/check_changelog_links.py --remote``, since a changelog heading
links to its tag before the tag is pushed. Network errors and timeouts count as failures after RETRIES
attempts. Standard library only.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = [
    "README.md",
    "CHANGELOG.md",
    "AGENTS.md",
    "docs/orgmode",
    "docs/newsfragments",
    "skills",
]
SUFFIXES = {".md", ".org", ".rst"}
CHALLENGE = {"eindir.rgoswami.me", "anneal.rgoswami.me"}
SKIP_HOSTS = {"localhost", "127.0.0.1", "example.com", "example.org"}
TAG_LINK = re.compile(r"^https://github\.com/HaoZeke/eindir/tree/v\d")
RETRIES = 3
TIMEOUT = 30
AGENT = "eindir-link-check/1 (+https://github.com/HaoZeke/eindir)"

URL = re.compile(r"https?://[^\s<>\"'`\]\[)(|{}]+")
DOI = re.compile(r"\b(10\.\d{4,9}/[^\s<>\"'`\]\[)(|,;{}]+)")
TRAILING = ".,;:!?*=~_"


def sources(args: list[str]) -> list[Path]:
    out = []
    for arg in args or DEFAULT:
        path = ROOT / arg
        if path.is_dir():
            out += sorted(p for p in path.rglob("*") if p.suffix in SUFFIXES)
        elif path.is_file():
            out.append(path)
    return out


def scan(paths: list[Path]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    urls: dict[str, set[str]] = {}
    dois: dict[str, set[str]] = {}
    for path in paths:
        rel = str(path.relative_to(ROOT))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            where = f"{rel}:{number}"
            for raw in URL.findall(line):
                url = raw.rstrip(TRAILING)
                host = urllib.parse.urlsplit(url).hostname or ""
                if host in SKIP_HOSTS or "{" in url or "$" in url or TAG_LINK.match(url):
                    continue
                if host in {"doi.org", "dx.doi.org"}:
                    path_part = urllib.parse.urlsplit(url).path.lstrip("/")
                    if path_part.startswith("doi/"):
                        path_part = path_part[4:]
                    dois.setdefault(urllib.parse.unquote(path_part), set()).add(where)
                    continue
                urls.setdefault(url, set()).add(where)
            stripped = URL.sub(" ", line)
            for raw in DOI.findall(stripped):
                dois.setdefault(raw.rstrip(TRAILING), set()).add(where)
    return urls, dois


def fetch(url: str) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, response.read(1 << 16)
    except urllib.error.HTTPError as error:
        return error.code, b""


def retry(check, target: str) -> str | None:
    problem = None
    for attempt in range(RETRIES):
        try:
            problem = check(target)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            problem = f"{type(error).__name__}: {error}"
        if problem is None or not problem.startswith(("URLError", "Timeout", "OSError")):
            return problem
        time.sleep(2**attempt)
    return problem


def check_url(url: str) -> str | None:
    status, _ = fetch(url)
    host = urllib.parse.urlsplit(url).hostname or ""
    if status < 400 or (status == 403 and host in CHALLENGE):
        return None
    if status == 429:
        raise TimeoutError("HTTP 429")
    return f"HTTP {status}"


def check_doi(doi: str) -> str | None:
    status, body = fetch("https://doi.org/api/handles/" + urllib.parse.quote(doi, safe="/"))
    if status == 404:
        return "DOI not registered"
    if status >= 400:
        raise TimeoutError(f"HTTP {status} from doi.org")
    code = json.loads(body).get("responseCode")
    return None if code == 1 else f"doi.org responseCode {code}"


def main(argv: list[str]) -> int:
    urls, dois = scan(sources(argv))
    jobs = [(check_doi, d, dois[d]) for d in sorted(dois)]
    jobs += [(check_url, u, urls[u]) for u in sorted(urls)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda job: retry(job[0], job[1]), jobs))
    failures = 0
    for (_, target, places), problem in zip(jobs, results):
        if problem:
            failures += 1
            print(f"{target}: {problem} ({', '.join(sorted(places))})")
    print(f"{len(dois)} DOI(s) and {len(urls)} URL(s) checked, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
