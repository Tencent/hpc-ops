# Copyright (C) 2026 Tencent.

import os
import shlex
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, version_macros=None, sourcedir=""):
        super().__init__(name, sources=[])
        self.version_macros = version_macros or []
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        check_build_environment()
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        build_lib_dir = os.path.dirname(self.get_ext_fullpath(ext.name))
        build_temp_dir = os.path.join(self.build_temp, ext.name)

        os.makedirs(build_lib_dir, exist_ok=True)
        os.makedirs(build_temp_dir, exist_ok=True)

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={build_lib_dir}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            *ext.version_macros,
            *get_extra_cmake_args(),
        ]

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp_dir)
        subprocess.check_call(
            ["cmake", "--build", ".", "--config", "Release", *get_cmake_parallel_args()],
            cwd=build_temp_dir,
        )

        so_src_path = os.path.join(build_temp_dir, "_C.abi3.so")
        so_dst_path = os.path.join(build_lib_dir, "hpc/_C.abi3.so")
        os.makedirs(os.path.dirname(so_dst_path), exist_ok=True)
        shutil.copy(so_src_path, so_dst_path)


def check_build_environment():
    if sys.version_info < (3, 9):
        raise RuntimeError("hpc-ops builds cp39 abi3 wheels and requires Python 3.9 or newer.")

    if shutil.which("cmake") is None:
        raise RuntimeError("cmake is required to build hpc-ops.")

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required in the active Python environment before building hpc-ops. "
            "Install the build requirements or run the wheel build with --no-isolation."
        ) from exc

    print(f"-- hpc-ops build Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"-- hpc-ops build PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")
    print(
        "-- hpc-ops build PyTorch CXX11 ABI: "
        f"{int(getattr(torch._C, '_GLIBCXX_USE_CXX11_ABI', True))}"
    )


def get_extra_cmake_args():
    return shlex.split(os.environ.get("HPC_OPS_CMAKE_ARGS", ""))


def get_build_parallelism():
    for env_name in ("CMAKE_BUILD_PARALLEL_LEVEL", "MAX_JOBS", "SLURM_CPUS_PER_TASK"):
        env_value = os.environ.get(env_name)
        if not env_value:
            continue

        try:
            jobs = int(env_value)
        except ValueError as exc:
            raise RuntimeError(f"{env_name} must be a positive integer, got {env_value!r}.") from exc

        if jobs < 1:
            raise RuntimeError(f"{env_name} must be a positive integer, got {env_value!r}.")

        return jobs

    return min(os.cpu_count() or 1, 16)


def get_cmake_parallel_args():
    return ["--parallel", str(get_build_parallelism())]


def get_version():
    git_hash = os.environ.get("HPC_GIT_HASH", "").strip()

    if not git_hash:
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "--short=7", "HEAD"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            git_hash = "unknown"

    if git_hash == "unknown":
        return "0.0.1.dev0", git_hash

    return f"0.0.1.dev0+g{git_hash}", git_hash


version, git_hash = get_version()
version_macros = [
    '-DHPC_VERSION_STR="{}"'.format(version),
    '-DHPC_GIT_HASH_STR="{}"'.format(git_hash),
]

with open("hpc/version.py", "w") as fp:
    fp.write('version = "{}"\n'.format(version))
    fp.write('git_hash = "{}"\n'.format(git_hash))

setup(
    name="hpc-ops",
    version=version,
    description="High Performance Computing Operator",
    author="Tencent hpc-ops authors",
    author_email="authors@hpc-ops",
    url="https://github.com/Tencent/hpc-ops",
    license="Copyright (C) 2026 Tencent.",
    packages=["hpc"],
    python_requires=">=3.9",
    ext_modules=[CMakeExtension("hpc", version_macros)],
    cmdclass={"build_ext": CMakeBuild},
    package_data={"hpc": ["*.so"]},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
    install_requires=["torch"],
)
