from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
import os

extra_compile_args = {
    'cxx': ['-O2', '-std=c++17', '-I./'],
    'nvcc': [
        '-arch=sm_90', '-O2', '-std=c++17', '--expt-relaxed-constexpr', '-I./'
    ]
}

cuda_extension = CUDAExtension(
    name='hpc._C',
    sources=[
        'src/_C.cc',
        'src/add/add.cu',
        'src/add/entry.cc',
        'src/cast/entry.cu',
    ],
    extra_compile_args=extra_compile_args)

setup(
    name='hpc',
    version='0.1.0',
    description='High Performance Computing Operator',
    ext_modules=[cuda_extension],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=True, parallel=1)
    },
    packages=['hpc'],
    package_data={"hpc": ["*.so"]},
    install_requires=['torch>=2.7.0'])
