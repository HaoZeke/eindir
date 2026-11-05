"""Build, lint and run the Rust examples in assets/ against eindir-core.

    python scripts/run_examples.py              # every example
    python scripts/run_examples.py objective_tour

The examples are binaries of a throwaway crate that depends on an eindir
checkout by path and starts from its Cargo.lock: this repository when
the skill sits in one, else $EINDIR_REPO, else a shallow clone of
https://github.com/HaoZeke/eindir. The ndarray and rand requirements
are read from the checkout's Cargo.toml so the types match.

Each example must pass rustfmt --check and clippy -D warnings. Every
rust block in SKILL.md must match a run of lines in one of them (up to
indentation), and the ndarray and rand versions in its toml block must
be those in Cargo.toml.

Needs Python 3.11 or newer and cargo for the rust-version in Cargo.toml.
Under rustup, an older default toolchain fails with "feature
`edition2024` is required"; run with RUSTUP_TOOLCHAIN=stable.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import tomllib
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]
ASSETS = SKILL / "assets"
REPO_URL = "https://github.com/HaoZeke/eindir"
SHARED_DEPENDENCIES = ("ndarray", "rand")


def is_checkout(path):
    cargo_toml = path / "Cargo.toml"
    if not cargo_toml.is_file():
        return False
    return tomllib.loads(cargo_toml.read_text()).get("package", {}).get("name") == "eindir-core"


def find_checkout(scratch):
    candidates = [SKILL.parents[1]]
    if os.environ.get("EINDIR_REPO"):
        candidates.append(Path(os.environ["EINDIR_REPO"]))
    for candidate in candidates:
        if is_checkout(candidate):
            return candidate.resolve()
    clone = scratch / "eindir"
    print(f"no eindir checkout found; cloning {REPO_URL}", flush=True)
    subprocess.run(["git", "clone", "--quiet", "--depth", "1", REPO_URL, str(clone)], check=True)
    return clone


def check_cargo(rust_version):
    out = subprocess.run(["cargo", "--version"], capture_output=True, text=True, check=True).stdout
    found = tuple(int(part) for part in re.search(r"(\d+)\.(\d+)", out).groups())
    wanted = tuple(int(part) for part in rust_version.split(".")[:2])
    if found < wanted:
        sys.exit(
            f"{out.strip()} is older than rust-version {rust_version} in Cargo.toml;"
            " with rustup, rerun with RUSTUP_TOOLCHAIN=stable"
        )


def manifest(checkout, package, examples):
    deps = package["dependencies"]
    lines = [
        "[package]",
        'name = "eindir-skill-examples"',
        'version = "0.0.0"',
        f'edition = "{package["package"]["edition"]}"',
        "publish = false",
        "",
        "[dependencies]",
        f"eindir-core = {{ path = {str(checkout)!r} }}",
    ]
    for name in SHARED_DEPENDENCIES:
        spec = deps[name]
        version = spec if isinstance(spec, str) else spec["version"]
        lines.append(f'{name} = "{version}"')
    for example in examples:
        lines += ["", "[[bin]]", f'name = "{example.stem}"', f"path = {str(example)!r}"]
    return "\n".join(lines) + "\n"


def blocks(text, language):
    return [m.group(1) for m in re.finditer(rf"^```{language}\n(.*?)^```$", text, re.M | re.S)]


def in_sources(block, sources):
    """True when the block, dedented, matches a run of lines in one source."""
    want = textwrap.dedent(block).strip("\n").splitlines()
    for lines in sources:
        for start in range(len(lines) - len(want) + 1):
            if textwrap.dedent("\n".join(lines[start : start + len(want)])).splitlines() == want:
                return True
    return False


def stale_snippets(skill_md, examples, package):
    """SKILL.md blocks that no longer match the examples or Cargo.toml."""
    sources = [e.read_text().splitlines() for e in examples]
    stale = [b for b in blocks(skill_md, "rust") if not in_sources(b, sources)]
    deps = package["dependencies"]
    for block in blocks(skill_md, "toml"):
        for name, spec in tomllib.loads(block).get("dependencies", {}).items():
            if name in SHARED_DEPENDENCIES:
                want = deps[name] if isinstance(deps[name], str) else deps[name]["version"]
                if spec != want:
                    stale.append(f'{name} = "{spec}" (Cargo.toml has "{want}")\n')
    return stale


def run(cmd, crate, env):
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=crate, env=env).returncode == 0


def main(names):
    every = sorted(ASSETS.glob("*.rs"))
    examples = [e for e in every if not names or e.stem in names]
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp)
        checkout = find_checkout(scratch)
        package = tomllib.loads((checkout / "Cargo.toml").read_text())
        for block in stale_snippets((SKILL / "SKILL.md").read_text(), every, package):
            print("SKILL.md block does not match assets/ or Cargo.toml:\n" + block)
            failures += 1
        check_cargo(package["package"]["rust-version"])
        crate = scratch / "examples"
        crate.mkdir()
        (crate / "Cargo.toml").write_text(manifest(checkout, package, examples))
        shutil.copy(checkout / "Cargo.lock", crate / "Cargo.lock")
        env = dict(os.environ, CARGO_TARGET_DIR=str(checkout / "target" / "skill-examples"))
        edition = package["package"]["edition"]
        failures += not run(["rustfmt", "--edition", edition, "--check", *map(str, examples)], crate, env)
        failures += not run(["cargo", "clippy", "--quiet", "--bins", "--", "-D", "warnings"], crate, env)
        for example in examples:
            failures += not run(["cargo", "run", "--quiet", "--bin", example.stem], crate, env)
    print(f"{failures} failure(s)" if failures else "all examples passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
