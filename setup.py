import os
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


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
        build_temp_dir = os.path.join(self.build_temp, ext.name)

        os.makedirs(build_lib_dir, exist_ok=True)
        os.makedirs(build_temp_dir, exist_ok=True)

        cmake_args = [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={build_lib_dir}", *ext.version_macros]

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp_dir)
        subprocess.check_call(
            ["cmake", "--build", ".", "--config", "Release", "-j16"], cwd=build_temp_dir
        )

        so_src_path = os.path.join(build_temp_dir, "_C.abi3.so")
        so_dst_path = os.path.join(build_lib_dir, "hpc/_C.abi3.so")
        shutil.copy(so_src_path, so_dst_path)


def get_version():
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "--short=7", "HEAD"], stderr=subprocess.DEVNULL, text=True
    ).strip()

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
    ext_modules=[CMakeExtension("hpc", version_macros)],
    cmdclass={"build_ext": CMakeBuild},
    package_data={"_C": ["*.so"]},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
    install_requires=["torch==2.7.0"],
)
