from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
import os

include_flags = '-I' + os.path.dirname(__file__)

extra_compile_args = {
    'cxx': ['-O2', '-std=c++17', include_flags],
    'nvcc': [
        '-arch=sm_90',
        '-O2',
        '-std=c++17',
        '--expt-relaxed-constexpr',
        include_flags,
    ]
}

extra_link_args = []

cuda_extension = CUDAExtension(
    name='hpc._C',
    sources=[
        'src/_C.cc',
        'src/add/add.cu',
        'src/add/entry.cc',
    ],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name='hpc',
    version='0.0.1',
    description='High Performance Computing Operator',
    packages=['hpc'],
    ext_modules=[cuda_extension],
    install_requires=['torch'],
    cmdclass={'build_ext': BuildExtension},
    package_data={"hpc": ["*.so"]},
    options={"bdist_wheel": {
        "py_limited_api": "cp39"
    }})
