import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import torch
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


def detect_sm_arch():
    """Detect SM architecture of the current GPU via torch."""
    try:
        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
        return str(major * 10 + minor)
    except Exception:
        return None


# ── Build mode ────────────────────────────────────────────────
# MULTI_ARCH=1       -> build sm90 + sm100 + sm103 into a single whl
# SM_ARCH=90         -> build sm90 only (legacy behaviour, backward compatible)
# SM_ARCH=100        -> build sm100 only
# neither set        -> auto-detect current GPU
MULTI_ARCH = os.environ.get("MULTI_ARCH", "0") == "1"

# Architecture list for multi-arch mode (extend as needed)
MULTI_ARCH_LIST = os.environ.get("MULTI_ARCH_LIST", "90,100,103").split(",")

if not MULTI_ARCH:
    SM_ARCH = os.environ.get("SM_ARCH")
    if not SM_ARCH:
        SM_ARCH = detect_sm_arch()
        if SM_ARCH:
            print(
                f"INFO: SM_ARCH not set, auto-detected from GPU: SM_ARCH={SM_ARCH}", file=sys.stderr
            )
        else:
            SM_ARCH = "90"
            print(
                f"WARNING: SM_ARCH not set and no GPU detected. Defaulting to SM_ARCH={SM_ARCH}.",
                file=sys.stderr,
            )


class CMakeExtension(Extension):
    def __init__(self, name, sm_arch, version_macros=[], sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sm_arch = sm_arch
        self.version_macros = version_macros
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        build_lib_dir = os.path.dirname(self.get_ext_fullpath(ext.name))

        # Separate build directory per architecture to avoid conflicts
        build_temp_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "build",
            f"sm{ext.sm_arch}",
            ext.name,
        )

        os.makedirs(build_lib_dir, exist_ok=True)
        os.makedirs(build_temp_dir, exist_ok=True)

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={build_lib_dir}",
            f"-DSM_ARCH={ext.sm_arch}",
            *ext.version_macros,
        ]

        # Ensure nvcc is discoverable if not in PATH
        cmake_env = os.environ.copy()

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=build_temp_dir, env=cmake_env
        )
        parallel = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "16")
        subprocess.check_call(
            ["cmake", "--build", ".", "--config", "Release", f"-j{parallel}"], cwd=build_temp_dir
        )

        # Filename with arch suffix: _C_sm90.abi3.so / _C_sm100.abi3.so
        so_filename = f"_C_sm{ext.sm_arch}.abi3.so"
        so_src_path = os.path.join(build_temp_dir, so_filename)
        so_dst_path = os.path.join(build_lib_dir, "hpc", so_filename)
        shutil.copy(so_src_path, so_dst_path)


def get_version():
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "--short=7", "HEAD"], stderr=subprocess.DEVNULL, text=True
    ).strip()

    newest_tag = (
        subprocess.check_output(
            ["git", "tag", "--sort=-v:refname"], stderr=subprocess.DEVNULL, text=True
        )
        .split("\n")[0]
        .strip()
        .lstrip("v")
    )
    torch_version = "torch" + torch.__version__.split("+")[0].replace(".", "")

    cuda_version = "cuda" + (torch.version.cuda or "").replace(".", "")

    # Arch suffix: multi-arch mode omits arch info; single-arch mode appends sm{arch}
    if MULTI_ARCH:
        arch_suffix = ""
    else:
        arch_suffix = f"sm{SM_ARCH}"

    newest_tag_hash = subprocess.check_output(
        ["git", "rev-list", "--tags", "--max-count=1"], stderr=subprocess.DEVNULL, text=True
    ).strip()[:7]
    if newest_tag_hash == git_hash:
        local = f"{torch_version}.{cuda_version}"
        if arch_suffix:
            local += f".{arch_suffix}"
        return f"{newest_tag}+{local}", git_hash
    else:
        try:
            commit_count = subprocess.check_output(
                ["git", "rev-list", "--count", f"{newest_tag_hash}..HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except subprocess.CalledProcessError:
            commit_count = "0"

        local = f"g{git_hash}.{torch_version}.{cuda_version}"
        if arch_suffix:
            local += f".{arch_suffix}"
        version_str = f"{newest_tag}.dev{commit_count}+{local}"
        return version_str, git_hash


version, git_hash = get_version()
version_macros = [
    "-DHPC_VERSION_STR={}".format(version),
    "-DHPC_GIT_HASH_STR={}".format(git_hash),
]

# Anchor version.py path relative to this script's location
_setup_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_setup_dir, "hpc", "version.py"), "w") as fp:
    fp.write('version = "{}"\n'.format(version))
    fp.write('git_hash = "{}"\n'.format(git_hash))

# ── Build extension list ──────────────────────────────────────
if MULTI_ARCH:
    # Multi-arch mode: create one CMakeExtension per architecture
    # Each will be cmake-built separately, producing _C_sm90.abi3.so / _C_sm100.abi3.so
    ext_modules = [
        CMakeExtension("hpc", sm_arch=arch, version_macros=version_macros)
        for arch in MULTI_ARCH_LIST
    ]
    print(f"INFO: MULTI_ARCH mode, building for: {MULTI_ARCH_LIST}", file=sys.stderr)
else:
    ext_modules = [CMakeExtension("hpc", sm_arch=SM_ARCH, version_macros=version_macros)]

# ── Description: include last commit id, author and message ──
try:
    _last_commit_id = subprocess.check_output(
        ["git", "rev-parse", "--short=7", "HEAD"], stderr=subprocess.DEVNULL, text=True
    ).strip()
    _last_commit_msg = subprocess.check_output(
        ["git", "log", "-1", "--pretty=%s"], stderr=subprocess.DEVNULL, text=True
    ).strip()
    _last_commit_author = subprocess.check_output(
        ["git", "log", "-1", "--pretty=%an"], stderr=subprocess.DEVNULL, text=True
    ).strip()
    description = f"{_last_commit_id} ({_last_commit_author}): {_last_commit_msg}"
except subprocess.CalledProcessError:
    description = "High Performance Computing Operator"

setup(
    name="hpc-ops",
    version=version,
    description=description,
    author="hpc-ops team",
    author_email="authors@hpc-ops",
    url="https://mirrors.tencent.com/#/private/pypi/detail?repo_id=155&project_name=hpc-ops",
    license="Copyright 2025",
    packages=find_packages(include=["hpc", "hpc.*", "dsl", "dsl.*"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuild},
    # Include the _C_sm*.abi3.so in the wheel.
    package_data={
        "hpc": [
            "*.so",
            "*.so.*",
        ],
        "dsl": ["**/*.txt"],
    },
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
    install_requires=[
        "torch",
        "nvidia-cutlass-dsl>=4.4",
    ],
)
