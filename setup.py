import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import torch
from setuptools import Extension, setup
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
        parallel = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "8")
        subprocess.check_call(
            ["cmake", "--build", ".", "--config", "Release", f"-j{parallel}"], cwd=build_temp_dir
        )

        # Filename with arch suffix: _C_sm90.abi3.so / _C_sm100.abi3.so
        so_filename = f"_C_sm{ext.sm_arch}.abi3.so"
        so_src_path = os.path.join(build_temp_dir, so_filename)
        so_dst_path = os.path.join(build_lib_dir, "hpc", so_filename)
        shutil.copy(so_src_path, so_dst_path)

        # Bundle every shared object produced by the NVSHMEM external project.
        # This must include not only libnvshmem_host.so[.3] (the NEEDED
        # dependency of _C_sm*.abi3.so) but also bootstrap plugins such as
        # nvshmem_bootstrap_uid.so.3, which NVSHMEM loads at runtime via
        # dlopen by bare filename
        # (see 3rd/ucl/nvshmem/src/host/bootstrap/bootstrap_loader.cpp).
        # Without these plugins the wheel will fail at test_communicator with
        #   "nvshmem_bootstrap_uid.so.3: cannot open shared object file".
        #
        # IMPORTANT (per-arch isolation):
        # libnvshmem_host.so statically links libnvshmem_device.a, whose
        # device fatbin is built per CUDA architecture (sm_90a / sm_100a /
        # sm_103a). Therefore libnvshmem_host.so produced under SM_ARCH=X
        # only contains a fatbin for sm_Xa and CANNOT be reused across
        # architectures. Running such a host lib on a GPU of a different
        # arch causes CUDA Runtime to fail registering the embedded module,
        # and downstream NVSHMEM calls fail with
        #   "Unable to access device state. 500"
        #   "Unable to access ibgda device state. 500"
        # which is exactly the failure mode of MULTI_ARCH=1 wheels prior to
        # this fix (see commit log for cf42b82).
        #
        # We therefore install the per-arch NVSHMEM shared libs into a
        # dedicated subdirectory (hpc/nvshmem/sm{ARCH}/) so each arch keeps
        # its own copy. hpc/__init__.py picks the right subdir at runtime
        # based on the detected GPU compute capability.
        #
        # NVSHMEM_INSTALL_DIR in CMakeLists.txt is set to
        #   ${CMAKE_CURRENT_BINARY_DIR}/nvshmem-install
        # i.e. <build_temp_dir>/nvshmem-install in this script.
        nvshmem_lib_dir = os.path.join(build_temp_dir, "nvshmem-install", "lib")
        hpc_pkg_dir = os.path.join(build_lib_dir, "hpc")
        nvshmem_arch_dir = os.path.join(hpc_pkg_dir, "nvshmem", f"sm{ext.sm_arch}")
        if os.path.isdir(nvshmem_lib_dir):
            os.makedirs(nvshmem_arch_dir, exist_ok=True)
            bundled = 0
            for entry in os.listdir(nvshmem_lib_dir):
                # Match libfoo.so / libfoo.so.X / libfoo.so.X.Y.Z and bare
                # plugin names like nvshmem_bootstrap_uid.so.3.
                if ".so" not in entry:
                    continue
                src = os.path.join(nvshmem_lib_dir, entry)
                # Skip directories (none expected, but be defensive).
                if os.path.isdir(src) and not os.path.islink(src):
                    continue
                dst = os.path.join(nvshmem_arch_dir, entry)
                # follow_symlinks=True (the default): wheels cannot contain
                # symlinks, so we materialise the symlink target as a real
                # file under its SONAME-style name (e.g. libnvshmem_host.so.3
                # -> real content of libnvshmem_host.so.3.x.y).
                shutil.copy(src, dst)
                bundled += 1
                print(f"INFO: bundled {entry} -> {dst}", file=sys.stderr)
            if bundled == 0:
                print(
                    f"WARNING: no NVSHMEM shared libs found under "
                    f"{nvshmem_lib_dir}; runtime will likely fail with "
                    f"'libnvshmem_host.so.3: cannot open shared object file'.",
                    file=sys.stderr,
                )
        else:
            print(
                f"WARNING: NVSHMEM install dir not found: {nvshmem_lib_dir}",
                file=sys.stderr,
            )


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

setup(
    name="hpc-ops",
    version=version,
    description="High Performance Computing Operator",
    author="hpc-ops team",
    author_email="authors@hpc-ops",
    url="https://mirrors.tencent.com/#/private/pypi/detail?repo_id=155&project_name=hpc-ops",
    license="Copyright 2025",
    packages=["hpc"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuild},
    # Important: include all .so files in the wheel.
    # "*.so.*" is required to pick up versioned SONAMEs such as
    # libnvshmem_host.so.3 which are otherwise excluded by "*.so".
    # The "nvshmem/sm*/..." patterns pick up per-arch NVSHMEM shared libs
    # bundled into hpc/nvshmem/sm{ARCH}/ subdirectories (see CMakeBuild
    # for the rationale on per-arch isolation).
    package_data={
        "hpc": [
            "*.so",
            "*.so.*",
            "nvshmem/sm*/*.so",
            "nvshmem/sm*/*.so.*",
        ]
    },
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
    install_requires=["torch"],
)
