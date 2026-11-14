"""Install the eindir C library into a prefix, then build and run the
examples in assets/ against it.

    python scripts/build_and_run.py             # every example
    python scripts/build_and_run.py raii_demo

The library comes from an eindir checkout: this repository when the
skill sits in one, else $EINDIR_REPO, else a shallow clone of
https://github.com/HaoZeke/eindir. cargo cinstall, from cargo-c, builds
it with the capi feature and installs the header, the libraries and
eindir-core.pc under target/skill-c-abi/prefix in the checkout.

C examples compile with $CC (default cc) as C99 and C++ examples with
$CXX (default c++) as C++17, both with -Wall -Wextra -Wpedantic -Werror
as in the ffi-smoke CI job. They take their flags from pkg-config when
it is on PATH and link to the shared library with an rpath. When
valgrind is on PATH every example runs under it as well.

The DLPack headers come from $DLPACK_ROOT/include when DLPACK_ROOT is
set, else from the GitHub archive of the DLPack release pinned in
.github/workflows/ci.yml, checked against the sha256 pinned there.

Every c and cpp block in SKILL.md must match a run of lines in one of
the examples, up to indentation.

Needs Python 3.11 or newer, cargo for the rust-version in Cargo.toml,
cargo-c at least package.metadata.capi.min_version, and C and C++
compilers. Under rustup, an older default toolchain fails with "feature
`edition2024` is required"; run with RUSTUP_TOOLCHAIN=stable.
"""

import hashlib
import io
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import tomllib
import urllib.request
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]
ASSETS = SKILL / "assets"
REPO_URL = "https://github.com/HaoZeke/eindir"
DLPACK_ARCHIVE = "https://github.com/dmlc/dlpack/archive/refs/tags/v{version}.tar.gz"
WARNINGS = ["-Wall", "-Wextra", "-Wpedantic", "-Werror"]
# suffix: (environment variable, default compiler, language standard, SKILL.md fence)
LANGUAGES = {
    ".c": ("CC", "cc", "-std=c99", "c"),
    ".cpp": ("CXX", "c++", "-std=c++17", "cpp"),
}
VALGRIND = [
    "valgrind",
    "--leak-check=full",
    "--show-leak-kinds=definite,indirect",
    "--track-origins=yes",
    "--error-exitcode=1",
]


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


def version_of(text):
    """The first dotted version in text as a tuple of three integers."""
    return tuple(int(p) for p in re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text).groups("0"))


def check_tools(package):
    rust_version = package["package"]["rust-version"]
    cargo = subprocess.run(["cargo", "--version"], capture_output=True, text=True, check=True)
    if version_of(cargo.stdout)[:2] < version_of(rust_version)[:2]:
        sys.exit(
            f"{cargo.stdout.strip()} is older than rust-version {rust_version} in Cargo.toml;"
            " with rustup, rerun with RUSTUP_TOOLCHAIN=stable"
        )
    minimum = package["package"]["metadata"]["capi"]["min_version"]
    cargo_c = subprocess.run(["cargo", "cinstall", "--version"], capture_output=True, text=True)
    if cargo_c.returncode != 0:
        sys.exit(f"cargo cinstall not found; install cargo-c {minimum} or newer: cargo install cargo-c")
    if version_of(cargo_c.stdout) < version_of(minimum):
        sys.exit(
            f"{cargo_c.stdout.strip()} is older than {minimum},"
            " package.metadata.capi.min_version in Cargo.toml"
        )


def fetch_dlpack(version, sha256, include):
    url = DLPACK_ARCHIVE.format(version=version)
    print(f"fetching {url}", flush=True)
    with urllib.request.urlopen(url) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        sys.exit(f"{url} has sha256 {digest}; .github/workflows/ci.yml pins {sha256}")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive.getmembers():
            parts = Path(member.name).parts
            if member.isfile() and parts[1:2] == ("include",) and ".." not in parts:
                target = include.joinpath(*parts[2:])
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())


def dlpack_include(checkout, work):
    ci = (checkout / ".github" / "workflows" / "ci.yml").read_text()
    version = re.search(r'DLPACK_VERSION:\s*"([^"]+)"', ci).group(1)
    if os.environ.get("DLPACK_ROOT"):
        include = Path(os.environ["DLPACK_ROOT"]) / "include"
    else:
        include = work / f"dlpack-{version}" / "include"
        if not (include / "dlpack" / "dlpack.h").is_file():
            sha256 = re.search(r"DLPACK_SHA256:\s*([0-9a-f]{64})", ci).group(1)
            fetch_dlpack(version, sha256, include)
    header = include / "dlpack" / "dlpack.h"
    if not header.is_file():
        sys.exit(f"{header} not found; set DLPACK_ROOT to a DLPack {version} source tree")
    found = re.search(r"#define DLPACK_MAJOR_VERSION (\d+)", header.read_text())
    if not found or found.group(1) != version.split(".")[0]:
        got = f"DLPack major version {found.group(1)}" if found else "a DLPack older than 1.0"
        sys.exit(f"{header} is {got}; eindir needs DLPack {version}, as pinned in ci.yml")
    return include


def compile_flags(prefix):
    """Compiler and linker flags for the installed eindir-core."""
    if not shutil.which("pkg-config"):
        print("pkg-config not on PATH; using the prefix layout directly")
        libdir = str(prefix / "lib")
        return [f"-I{prefix / 'include'}"], [f"-L{libdir}", "-leindir_core", f"-Wl,-rpath,{libdir}"]
    env = {k: v for k, v in os.environ.items() if not k.startswith("PKG_CONFIG")}
    env["PKG_CONFIG_PATH"] = str(prefix / "lib" / "pkgconfig")

    def query(option):
        cmd = ["pkg-config", option, "eindir-core"]
        return shlex.split(subprocess.run(cmd, env=env, capture_output=True, text=True, check=True).stdout)

    libdir = query("--variable=libdir")[0]
    return query("--cflags"), [*query("--libs"), f"-Wl,-rpath,{libdir}"]


def blocks(text, fence):
    return [m.group(1) for m in re.finditer(rf"^```{fence}\n(.*?)^```$", text, re.M | re.S)]


def in_sources(block, sources):
    """True when the block, dedented, matches a run of lines in one source."""
    want = textwrap.dedent(block).strip("\n").splitlines()
    for lines in sources:
        for start in range(len(lines) - len(want) + 1):
            if textwrap.dedent("\n".join(lines[start : start + len(want)])).splitlines() == want:
                return True
    return False


def stale_snippets(skill_md, examples):
    """SKILL.md blocks that no longer match an example."""
    stale = []
    for suffix, (*_, fence) in LANGUAGES.items():
        sources = [e.read_text().splitlines() for e in examples if e.suffix == suffix]
        stale += [b for b in blocks(skill_md, fence) if not in_sources(b, sources)]
    return stale


def run(cmd, cwd=None, env=None):
    print("$ " + shlex.join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, env=env).returncode == 0


def main(names):
    every = sorted(p for p in ASSETS.iterdir() if p.suffix in LANGUAGES)
    unknown = set(names) - {e.stem for e in every}
    if unknown:
        sys.exit(f"no example named {', '.join(sorted(unknown))} in {ASSETS}")
    examples = [e for e in every if not names or e.stem in names]
    failures = 0
    for block in stale_snippets((SKILL / "SKILL.md").read_text(), every):
        print("SKILL.md block does not match assets/:\n" + block)
        failures += 1
    with tempfile.TemporaryDirectory() as tmp:
        checkout = find_checkout(Path(tmp))
        package = tomllib.loads((checkout / "Cargo.toml").read_text())
        check_tools(package)
        work = checkout / "target" / "skill-c-abi"
        work.mkdir(parents=True, exist_ok=True)
        include = dlpack_include(checkout, work)
        prefix = work / "prefix"
        shutil.rmtree(prefix, ignore_errors=True)
        env = dict(os.environ, CARGO_TARGET_DIR=str(work))
        install = ["cargo", "cinstall", "--release", f"--prefix={prefix}", "--libdir=lib"]
        if not run(install, cwd=checkout, env=env):
            return 1
        cflags, libs = compile_flags(prefix)
        bin_dir = work / "bin"
        bin_dir.mkdir(exist_ok=True)
        valgrind = shutil.which("valgrind")
        if not valgrind:
            print("valgrind not on PATH; the examples run without it")
        for example in examples:
            variable, default, standard, _ = LANGUAGES[example.suffix]
            compiler = shlex.split(os.environ.get(variable, default))
            binary = bin_dir / example.stem
            source = [standard, *WARNINGS, f"-I{include}", *cflags, str(example)]
            if not run([*compiler, *source, *libs, "-o", str(binary)]):
                failures += 1
                continue
            failures += not run([str(binary)])
            if valgrind:
                failures += not run([*VALGRIND, "--quiet", str(binary)])
    print(f"{failures} failure(s)" if failures else "all examples passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
