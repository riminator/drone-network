"""
install.py
Cross-platform setup script for the drone-network project.

Handles:
  - Core Python packages (gymnasium, torch, pyyaml, numpy)
  - pybullet source build with macOS/clang patches (macOS only)
  - pybullet pre-built wheel install (Windows / Linux)
  - gym-pybullet-drones from GitHub

Supported platforms:
  macOS   arm64 / x86_64   ✓  (patches clang21 + SDK26 build issues)
  Windows x86_64           ✓  (uses pre-built pybullet wheel)
  Linux   x86_64 / arm64   ✓  (standard pip install)

Usage:
    python install.py              # full install
    python install.py --check      # verify everything is installed correctly
    python install.py --skip-torch # skip PyTorch (e.g. you have a custom CUDA build)
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], check: bool = True, env: dict | None = None, **kwargs):
    """Run a command, streaming output, optionally raising on failure."""
    merged_env = {**os.environ, **(env or {})}
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, env=merged_env, **kwargs)
    if check and result.returncode != 0:
        print(f"\n[ERROR] Command failed (exit {result.returncode}): {' '.join(cmd)}")
        sys.exit(result.returncode)
    return result


def pip(*args):
    """Run pip via the current Python interpreter (avoids PATH confusion)."""
    run([sys.executable, "-m", "pip", "install", "--no-cache-dir", *args])


def pip_check(*packages) -> bool:
    """Return True only if all packages are importable."""
    for pkg in packages:
        result = subprocess.run(
            [sys.executable, "-c", f"import {pkg}"],
            capture_output=True,
        )
        if result.returncode != 0:
            return False
    return True


def header(msg: str):
    bar = "─" * 60
    print(f"\n{bar}\n  {msg}\n{bar}")


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

SYSTEM   = platform.system()    # 'Darwin' | 'Windows' | 'Linux'
MACHINE  = platform.machine()   # 'arm64' | 'x86_64' | 'AMD64'
PY_VER   = sys.version_info

IS_MAC     = SYSTEM == "Darwin"
IS_WIN     = SYSTEM == "Windows"
IS_LINUX   = SYSTEM == "Linux"
IS_ARM_MAC = IS_MAC and MACHINE == "arm64"

# Clang version — only meaningful on macOS
_CLANG_MAJOR = 0
if IS_MAC:
    try:
        import re as _re
        out = subprocess.check_output(["clang", "--version"], stderr=subprocess.STDOUT, text=True)
        # e.g. "Apple clang version 21.0.0 (clang-2100.1.1.101)"
        m = _re.search(r"version\s+(\d+)", out)
        if m:
            _CLANG_MAJOR = int(m.group(1))
    except Exception:
        pass

# macOS SDK version — check for the problematic SDK 26 (Tahoe)
_SDK_MAJOR = 0
if IS_MAC:
    try:
        sdk_path_out = subprocess.check_output(
            ["xcrun", "--show-sdk-path"], text=True
        ).strip()
        sdk_ver_out = subprocess.check_output(
            ["xcrun", "--show-sdk-version"], text=True
        ).strip()
        _SDK_MAJOR = int(sdk_ver_out.split(".")[0])
    except Exception:
        pass

NEEDS_SOURCE_BUILD = IS_MAC and (_CLANG_MAJOR >= 17 or _SDK_MAJOR >= 15)


# ---------------------------------------------------------------------------
# Step 1 — Core packages
# ---------------------------------------------------------------------------

def install_core(skip_torch: bool):
    header("Step 1 — Core packages")

    core = [
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
    ]
    pip(*core)

    if not skip_torch:
        header("Step 1b — PyTorch")
        if IS_WIN:
            # Point at the official PyTorch Windows index for CUDA/CPU wheels
            print("Installing PyTorch for Windows (CPU build).")
            print("For CUDA, visit https://pytorch.org/get-started/locally/ and install manually.")
            pip("torch", "--index-url", "https://download.pytorch.org/whl/cpu")
        else:
            pip("torch>=2.0.0")
    else:
        print("Skipping PyTorch (--skip-torch flag set).")


# ---------------------------------------------------------------------------
# Step 2 — pybullet
# ---------------------------------------------------------------------------

def install_pybullet():
    header("Step 2 — pybullet")

    if pip_check("pybullet"):
        print("pybullet already installed — skipping.")
        return

    if IS_WIN or IS_LINUX:
        # Pre-built binary wheels exist for Windows and Linux — no compilation needed
        print(f"Installing pybullet pre-built wheel for {SYSTEM}.")
        pip("pybullet")
        return

    # macOS: binary wheel only exists for older SDK/clang combos.
    # Try the wheel first; if it fails, fall back to patched source build.
    print(f"macOS detected (clang {_CLANG_MAJOR}, SDK {_SDK_MAJOR}).")

    if not NEEDS_SOURCE_BUILD:
        print("Clang/SDK version looks compatible — trying binary wheel.")
        result = run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "pybullet"],
            check=False,
        )
        if result.returncode == 0:
            print("Binary wheel installed successfully.")
            return
        print("Binary wheel failed — falling back to patched source build.")

    _install_pybullet_from_source()


def _install_pybullet_from_source():
    """
    Download pybullet source, apply two patches for macOS SDK 15+/26 / clang 17+,
    then build and install.

    Patch 1 — setup.py:
        Adds -Wno-deprecated-declarations and -Wno-incompatible-function-pointer-types
        to CXX_FLAGS so clang does not treat sprintf deprecation as a hard error.

    Patch 2 — examples/ThirdPartyLibs/zlib/zutil.h:
        Removes the '#define fdopen(fd, mode) NULL' macro on Apple platforms.
        Pybullet's vendored zlib redefines fdopen as NULL on MACOS, which then
        collides with the real fdopen declaration in macOS SDK 26's _stdio.h,
        causing clang 21 to emit a parse error.
    """
    header("Step 2 (source build) — pybullet with macOS patches")

    # Locate or download source tarball
    import urllib.request

    PYBULLET_VERSION = "3.2.7"
    TARBALL_URL = (
        f"https://files.pythonhosted.org/packages/source/p/pybullet/"
        f"pybullet-{PYBULLET_VERSION}.tar.gz"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tarball = tmpdir / f"pybullet-{PYBULLET_VERSION}.tar.gz"

        print(f"Downloading pybullet {PYBULLET_VERSION} source...")
        urllib.request.urlretrieve(TARBALL_URL, tarball)

        import tarfile as tarmod
        with tarmod.open(tarball, "r:gz") as tf:
            tf.extractall(tmpdir)

        src_dir = tmpdir / f"pybullet-{PYBULLET_VERSION}"

        # ---- Patch 1: setup.py — suppress clang warnings-as-errors ----
        _patch_setup_py(src_dir / "setup.py")

        # ---- Patch 2: zutil.h — stop fdopen macro poisoning stdio.h ----
        _patch_zutil_h(src_dir / "examples" / "ThirdPartyLibs" / "zlib" / "zutil.h")

        # Build and install from patched source
        print("Building pybullet from patched source (this takes ~2 minutes)...")
        run([sys.executable, "-m", "pip", "install", str(src_dir), "--no-cache-dir"])

    print("pybullet installed from source ✓")


def _patch_setup_py(path: Path):
    src = path.read_text()
    old = "CXX_FLAGS += '-DBT_ENABLE_VHACD '\n\nEGL_CXX_FLAGS"
    new = (
        "CXX_FLAGS += '-DBT_ENABLE_VHACD '\n"
        "# macOS clang 17+ / SDK 15+: sprintf is deprecated in the new SDK;\n"
        "# suppress warnings that clang promotes to errors.\n"
        "CXX_FLAGS += '-Wno-deprecated-declarations '\n"
        "CXX_FLAGS += '-Wno-incompatible-function-pointer-types '\n"
        "\nEGL_CXX_FLAGS"
    )
    if old not in src:
        print("[WARN] setup.py pattern not found — skipping Patch 1 (may still work)")
        return
    path.write_text(src.replace(old, new, 1))
    print("  Patch 1 applied: setup.py CXX_FLAGS ✓")


def _patch_zutil_h(path: Path):
    src = path.read_text()
    old = "#ifndef fdopen\n#define fdopen(fd, mode) NULL /* No fdopen() */\n#endif"
    new = (
        "#ifndef fdopen\n"
        "/* macOS 10.6+ ships fdopen in <stdio.h>. Redefining it as NULL\n"
        "   causes a parse error in macOS SDK 15+ (_stdio.h:322). Skip on Apple. */\n"
        "#if !defined(__APPLE__)\n"
        "#define fdopen(fd, mode) NULL /* No fdopen() */\n"
        "#endif /* !__APPLE__ */\n"
        "#endif"
    )
    if old not in src:
        print("[WARN] zutil.h pattern not found — skipping Patch 2 (may still work)")
        return
    path.write_text(src.replace(old, new, 1))
    print("  Patch 2 applied: zutil.h fdopen guard ✓")


# ---------------------------------------------------------------------------
# Step 3 — gym-pybullet-drones
# ---------------------------------------------------------------------------

def install_gym_pybullet_drones():
    header("Step 3 — gym-pybullet-drones")

    if pip_check("gym_pybullet_drones"):
        print("gym-pybullet-drones already installed — skipping.")
        return

    pip("git+https://github.com/utiasDSL/gym-pybullet-drones.git")


# ---------------------------------------------------------------------------
# Step 4 — Optional W&B
# ---------------------------------------------------------------------------

def install_optional():
    header("Step 4 — Optional packages")
    print("Installing wandb (optional — safe to skip if unwanted).")
    result = run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", "wandb"],
        check=False,
    )
    if result.returncode != 0:
        print("[WARN] wandb install failed — set use_wandb: false in training/config.yaml.")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify():
    header("Verification")
    checks = {
        "numpy":              "import numpy; print('  numpy', numpy.__version__)",
        "gymnasium":          "import gymnasium; print('  gymnasium', gymnasium.__version__)",
        "torch":              "import torch; print('  torch', torch.__version__)",
        "pybullet":           "import pybullet; print('  pybullet OK')",
        "gym_pybullet_drones":"from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary; print('  gym-pybullet-drones OK')",
        "project envs":       "import sys; sys.path.insert(0,'.'); from envs.pybullet_env import PybulletHomeEnv, _PYBULLET_AVAILABLE; print('  PybulletHomeEnv, pybullet_available =', _PYBULLET_AVAILABLE)",
        "project lab":        "import sys; sys.path.insert(0,'.'); from lab.deploy import deploy; print('  lab.deploy OK')",
    }

    all_ok = True
    for name, snippet in checks.items():
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"✓  {name}: {result.stdout.strip()}")
        else:
            print(f"✗  {name}: FAILED\n   {result.stderr.strip()[:200]}")
            all_ok = False

    print()
    if all_ok:
        print("All checks passed — you're ready to train and deploy.")
        print("\nTo start training:")
        print("  python -m training.train_mappo")
        print("\nTo deploy a checkpoint:")
        print("  python -m lab.deploy --checkpoint checkpoints/<file>.pt")
    else:
        print("Some checks failed — review the errors above.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Windows compatibility notes (printed, not enforced)
# ---------------------------------------------------------------------------

WINDOWS_NOTES = """
Windows Compatibility Notes
────────────────────────────────────────────────────────────
✓  pybullet    — pre-built wheels exist for Windows x86_64.
                  No source compilation needed.

✓  PyTorch     — CPU build installed automatically.
                  For CUDA: https://pytorch.org/get-started/locally/

✓  Training    — HomeEnv (envs/home_env.py) runs identically.
                  Training does NOT need pybullet at all.

⚠  PyBullet GUI — works on Windows but requires a display.
                  No changes needed; the GUI window will open.

⚠  gym-pybullet-drones — pip-installable from GitHub on Windows,
                  but requires Git in PATH:
                    winget install Git.Git
                  or install from zip:
                    pip install https://github.com/utiasDSL/gym-pybullet-drones/archive/main.zip

⚠  Multiprocessing — Ray RLlib (optional, not required by this project)
                  has known issues on Windows with spawn-mode.
                  The built-in train_mappo.py does NOT use Ray — no issue.

✓  Everything else (numpy, gymnasium, yaml, wandb) — normal pip install.
────────────────────────────────────────────────────────────
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"\nDrone Network — Installer")
    print(f"Python {sys.version.split()[0]}  |  {SYSTEM} {MACHINE}")
    print(f"macOS clang {_CLANG_MAJOR}, SDK {_SDK_MAJOR}" if IS_MAC else "")

    if IS_WIN:
        print(WINDOWS_NOTES)

    parser = argparse.ArgumentParser(description="Install drone-network dependencies")
    parser.add_argument("--check",      action="store_true", help="Only verify installs, do not install")
    parser.add_argument("--skip-torch", action="store_true", help="Skip PyTorch install")
    parser.add_argument("--skip-wandb", action="store_true", help="Skip optional wandb install")
    args = parser.parse_args()

    if args.check:
        verify()
        return

    # Need git for gym-pybullet-drones
    if not shutil.which("git"):
        print(
            "\n[ERROR] git is not in PATH.\n"
            "  macOS:   xcode-select --install\n"
            "  Windows: winget install Git.Git  (then restart terminal)\n"
            "  Ubuntu:  sudo apt install git\n"
        )
        sys.exit(1)

    install_core(skip_torch=args.skip_torch)
    install_pybullet()
    install_gym_pybullet_drones()
    if not args.skip_wandb:
        install_optional()

    verify()


if __name__ == "__main__":
    main()
