from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
import os
from glob import glob
import subprocess


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
    newest_tag_hash = subprocess.check_output(
        ["git", "rev-list", "--tags", "--max-count=1"], stderr=subprocess.DEVNULL, text=True
    ).strip()[:7]
    if newest_tag_hash == git_hash:
        return newest_tag
    else:
        try:
            commit_count = subprocess.check_output(
                ["git", "rev-list", "--count", f"{newest_tag_hash}..HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except subprocess.CalledProcessError:
            commit_count = "0"

        return f"{newest_tag}.dev{commit_count}+g{git_hash}"


version = get_version()

include_flags = "-I" + os.path.dirname(__file__)
cute_include = "-I" + os.path.dirname(__file__) + "/3rd/cutlass/include"
cxx_flags = '-DHPC_VERSION_STR="{}"'.format(version)

extra_compile_args = {
    "cxx": ["-O2", "-std=c++17", include_flags, cute_include, cxx_flags],
    "nvcc": [
        "-arch=sm_90a",
        "-O2",
        "-lineinfo",
        "-Xptxas",
        "-v",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
        include_flags,
        cute_include,
    ],
}

extra_link_args = []

cc_files = glob("src/**/*.cc", recursive=True)
cu_files = glob("src/**/*.cu", recursive=True)

sources = cc_files + cu_files
sources = [f for f in sources if not ("test" in f)]

cuda_extension = CUDAExtension(
    name="hpc._C",
    sources=sources,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    py_limited_api=True,
)


setup(
    name="hpc-ops",
    version=version,
    description="High Performance Computing Operator",
    packages=["hpc"],
    ext_modules=[cuda_extension],
    install_requires=["torch"],
    cmdclass={"build_ext": BuildExtension},
    package_data={"hpc": ["*.so"]},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
