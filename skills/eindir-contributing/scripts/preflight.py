"""Check a change to eindir before it is committed or reviewed.

    python scripts/preflight.py                      # the working tree
    python scripts/preflight.py --base origin/main   # and the commits since
    python scripts/preflight.py --subject "fix(ffi): reject NULL bounds"

Each check prints ok, FAIL or skip:

- fragments: every file in the towncrier directory sits at its top
  level, names a type from towncrier.toml, starts with + or an issue
  number, and is not empty. towncrier skips or drops the others without
  a warning, or links an issue named after the file;
- llms: llms.txt and llms-full.txt match the sources
  (scripts/llms/build_llms.py --check);
- version: every copy of the version matches Cargo.toml
  (tools/check-version-drift.sh). Skipped when this Python cannot import
  cogapp or cargo is older than the rust-version in Cargo.toml;
- subjects: with --base, each commit subject since the merge base, and
  the --subject text, starts with a type that cog accepts and does not
  end with a period;
- fragment needed: with --base, a change outside docs/ since the merge
  base, committed or not, comes with a new fragment.

Run it from inside the eindir checkout that holds this skill, or from
any directory of another eindir checkout. Needs Python 3.11 or newer
and git.
"""

import argparse
import importlib.util
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]
# The types cocogitto accepts without configuration; cog.toml adds more.
COG_DEFAULT_TYPES = ("build", "chore", "ci", "docs", "feat", "fix", "perf", "refactor", "revert", "style", "test")
SUBJECT = re.compile(r"^(?P<type>[A-Za-z]+)(\([^()]+\))?!?: \S")
results = []


def report(status, check, detail):
    results.append(status)
    print(f"{status:4} {check}: {detail}", flush=True)


def is_checkout(path):
    cargo_toml = path / "Cargo.toml"
    if not cargo_toml.is_file():
        return False
    return tomllib.loads(cargo_toml.read_text()).get("package", {}).get("name") == "eindir-core"


def find_checkout():
    if is_checkout(SKILL.parents[1]):
        return SKILL.parents[1]
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if top.returncode == 0 and is_checkout(Path(top.stdout.strip())):
        return Path(top.stdout.strip())
    sys.exit("not inside an eindir checkout")


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def towncrier(repo):
    config = tomllib.loads((repo / "towncrier.toml").read_text())["tool"]["towncrier"]
    return repo / config["directory"], [t["directory"] for t in config["type"]]


def fragment_problem(path, directory, types):
    """Why towncrier would skip, drop or mislink a fragment, or None."""
    if path.parent != directory:
        return "is in a subdirectory, which towncrier does not read"
    # towncrier takes the last dot-separated part that names a type; the
    # parts before it are the issue, and a + prefix means no issue.
    parts = path.name.split(".")
    kinds = [i for i in range(1, len(parts)) if parts[i] in types]
    if not kinds:
        return f"names no type ({', '.join(types)}), so towncrier skips it"
    issue = ".".join(parts[: kinds[-1]])
    if not (issue.startswith("+") or issue.isdigit()):
        return f"would link an issue named {issue!r}; start the name with + or an issue number"
    if not path.read_text().strip():
        return "is empty, so towncrier drops it"
    return None


def check_fragments(repo):
    directory, types = towncrier(repo)
    fragments, bad = [], []
    for path in sorted(directory.rglob("*")):
        if path.is_dir() or path.name == ".gitkeep":
            continue
        problem = fragment_problem(path, directory, types)
        if problem:
            bad.append(f"{path.relative_to(repo)} {problem}")
        else:
            fragments.append(path)
    for problem in bad:
        report("FAIL", "fragments", problem)
    if not bad:
        report("ok", "fragments", f"{len(fragments)} pending")


def check_llms(repo):
    generator = repo / "scripts" / "llms" / "build_llms.py"
    if not generator.is_file():
        report("skip", "llms", f"no {generator.relative_to(repo)}")
        return
    done = subprocess.run([sys.executable, str(generator), "--check"], cwd=repo, capture_output=True, text=True)
    if done.returncode == 0:
        report("ok", "llms", "llms.txt and llms-full.txt are current")
    else:
        report("FAIL", "llms", (done.stdout + done.stderr).strip())


def cargo_too_old(repo):
    rust_version = tomllib.loads((repo / "Cargo.toml").read_text())["package"]["rust-version"]
    try:
        out = subprocess.run(["cargo", "--version"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return "cargo not found"
    found = tuple(int(p) for p in re.search(r"(\d+)\.(\d+)", out).groups())
    if found < tuple(int(p) for p in rust_version.split(".")[:2]):
        return (
            f"{out.strip()} is older than rust-version {rust_version}, and the script would"
            " blame Cargo.lock; with rustup, rerun with RUSTUP_TOOLCHAIN=stable"
        )
    return None


def check_version(repo):
    if importlib.util.find_spec("cogapp") is None:
        report("skip", "version", f"{sys.executable} cannot import cogapp (pip install cogapp)")
        return
    reason = cargo_too_old(repo)
    if reason:
        report("skip", "version", reason)
        return
    env = dict(os.environ, PYTHON=sys.executable)
    done = subprocess.run(["bash", "tools/check-version-drift.sh"], cwd=repo, env=env, capture_output=True, text=True)
    if done.returncode == 0:
        report("ok", "version", "every copy matches Cargo.toml")
    else:
        report("FAIL", "version", (done.stdout + done.stderr).strip())


def commit_types(repo):
    custom = tomllib.loads((repo / "cog.toml").read_text()).get("commit_types", {})
    return sorted(set(COG_DEFAULT_TYPES) | set(custom))


def subject_problem(subject, types):
    match = SUBJECT.match(subject)
    if not match:
        return "is not 'type(scope): description' or 'type: description'"
    kind = match.group("type")
    if kind.lower() not in types:
        return f"has type {kind!r}, which cog rejects"
    if kind != kind.lower():
        return f"has type {kind!r}; commits since early 2024 write the type in lowercase"
    if subject.rstrip().endswith("."):
        return "ends with a period"
    return None


def check_subjects(repo, subjects):
    types = commit_types(repo)
    bad = [(s, problem) for s in subjects if (problem := subject_problem(s, types))]
    for subject, problem in bad:
        report("FAIL", "subjects", f"{subject!r} {problem}")
    if bad:
        print(f"     types cog accepts here: {', '.join(types)}")
    else:
        report("ok", "subjects", f"{len(subjects)} checked")


def check_fragment_needed(repo, base):
    directory, _ = towncrier(repo)
    fragments = directory.relative_to(repo).as_posix() + "/"
    changed = git(repo, "diff", "--name-only", base).splitlines()
    untracked = git(repo, "ls-files", "--others", "--exclude-standard").splitlines()
    added = git(repo, "diff", "--name-only", "--diff-filter=A", base, "--", fragments).splitlines()
    added += [p for p in untracked if p.startswith(fragments)]
    outside = [p for p in changed + untracked if not p.startswith("docs/")]
    if outside and not added:
        shown = ", ".join(outside[:5]) + (", ..." if len(outside) > 5 else "")
        report("FAIL", "fragment needed", f"{shown} changed and no fragment was added under {fragments}")
    else:
        report("ok", "fragment needed", f"{len(outside)} file(s) outside docs/, {len(added)} new fragment(s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", help="branch or commit the change starts from, such as origin/main")
    parser.add_argument("--subject", action="append", default=[], help="a commit subject to check")
    args = parser.parse_args()
    repo = find_checkout()
    check_fragments(repo)
    check_llms(repo)
    check_version(repo)
    subjects = list(args.subject)
    if args.base:
        merge_base = git(repo, "merge-base", args.base, "HEAD").strip()
        subjects += git(repo, "log", "--no-merges", "--format=%s", f"{merge_base}..HEAD").splitlines()
        check_fragment_needed(repo, merge_base)
    if args.base or args.subject:
        check_subjects(repo, subjects)
    failed = results.count("FAIL")
    print(f"{failed} check(s) failed" if failed else "preflight passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
