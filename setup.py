# Copyright (C) 2026 Tencent.

import os
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# Architectures the build supports. The same list is in the Makefile and
# CMakeLists.txt; adding one also needs an HPC_ARCH_EVAL_<arch> block in
# src/utils/utils.h.
KNOWN_ARCHS = [90, 100, 103, 120]


class CMakeExtension(Extension):
    def __init__(self, name, version_macros=[], sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.version_macros = version_macros
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        build_lib_dir = os.path.dirname(self.get_ext_fullpath(ext.name))
        so_dst_dir = os.path.join(build_lib_dir, "hpc")
        os.makedirs(so_dst_dir, exist_ok=True)

        # One module per architecture, each from its own CMake tree, so `make
        # sm90` then `make sm103` then `make sm90` again reuses the sm90 object
        # files instead of overwriting a shared build/temp directory. The wheel
        # ends up with every module; hpc/__init__.py picks one at import time.
        for arch in sm_archs:
            build_temp_dir = os.path.join(ext.sourcedir, "build", f"sm{arch}")
            os.makedirs(build_temp_dir, exist_ok=True)

            cmake_args = [
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={build_temp_dir}",
                f"-DPython3_EXECUTABLE={sys.executable}",
                f"-DHPC_TARGET_ARCH={arch}",
                *ext.version_macros,
            ]

            subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp_dir)
            subprocess.check_call(
                ["cmake", "--build", ".", "--config", "Release", "-j64"], cwd=build_temp_dir
            )

            so_name = f"_C_sm{arch}.abi3.so"
            shutil.copy(os.path.join(build_temp_dir, so_name), os.path.join(so_dst_dir, so_name))

        # A module left over from an earlier build with a different SM_ARCH would
        # be picked up at import time, so the output directory must contain
        # exactly what this build produced.
        kept = {f"_C_sm{arch}.abi3.so" for arch in sm_archs}
        for name in os.listdir(so_dst_dir):
            if name.startswith("_C_sm") and name.endswith(".abi3.so") and name not in kept:
                os.remove(os.path.join(so_dst_dir, name))


def get_version():
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "--short=7", "HEAD"], stderr=subprocess.DEVNULL, text=True
    ).strip()

    return f"0.0.1.dev0+g{git_hash}", git_hash


def get_sm_archs():
    """Target architectures, from SM_ARCH if set, otherwise the local GPUs'."""
    sm_arch = os.getenv("SM_ARCH", "").strip()
    if not sm_arch:
        caps = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], text=True
        )
        # "9.0" -> 90, "10.3" -> 103.
        sm_arch = ",".join(cap.strip().replace(".", "") for cap in caps.split())
        if not sm_arch:
            raise SystemExit(
                "No CUDA device detected. Set the target explicitly, "
                "e.g. 'SM_ARCH=90 make' or 'make sm90'."
            )
    parts = [p.strip() for p in sm_arch.split(",") if p.strip()]
    if not parts or any(not p.isdigit() or int(p) not in KNOWN_ARCHS for p in parts):
        raise SystemExit(
            f"SM_ARCH={sm_arch!r} is not a comma separated list of {KNOWN_ARCHS}, "
            "e.g. 'SM_ARCH=90 make' or 'SM_ARCH=90,103 make wheel'."
        )
    return sorted({int(p) for p in parts})


version, git_hash = get_version()
sm_archs = get_sm_archs()

# A wheel only runs on the architectures it has a module for, so the version has
# to say which — two wheels of the same public version are not interchangeable.
# Omitted when the build covers every architecture KNOWN_ARCHS lists in the
# Makefile. Examples: ".sm90" (one arch), ".sm90.100.103" (a subset).
if sm_archs != KNOWN_ARCHS:
    version += ".sm" + ".".join(str(a) for a in sm_archs)

version_macros = [
    '-DHPC_VERSION_STR="{}"'.format(version),
    '-DHPC_GIT_HASH_STR="{}"'.format(git_hash),
]

with open("hpc/version.py", "w") as fp:
    fp.write('version = "{}"\n'.format(version))
    fp.write('git_hash = "{}"\n'.format(git_hash))

bdist_wheel_options = {"py_limited_api": "cp39"}

setup(
    name="hpc-ops",
    version=version,
    description="High Performance Computing Operator",
    author="Tencent hpc-ops authors",
    author_email="authors@hpc-ops",
    url="https://github.com/Tencent/hpc-ops",
    license="Copyright (C) 2026 Tencent.",
    packages=["hpc"],
    ext_modules=[CMakeExtension("hpc", version_macros)],
    cmdclass={"build_ext": CMakeBuild},
    package_data={"hpc": ["*.so"]},
    options={"bdist_wheel": bdist_wheel_options},
    install_requires=["torch"],
)
