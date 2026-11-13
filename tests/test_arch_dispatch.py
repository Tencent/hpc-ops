import os
import shlex
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("arch", [90, 100, 103, 120])
def test_arch_dispatch(arch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    binary = tmp_path / f"arch_dispatch_sm{arch}"
    subprocess.run(
        [
            *shlex.split(os.environ.get("CXX", "c++")),
            "-std=c++20",
            "-O0",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-DHPC_TARGET_ARCH={arch}",
            "-I",
            str(root),
            str(root / "tests/arch_dispatch.cc"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
