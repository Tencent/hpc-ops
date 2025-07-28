from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
import os
from glob import glob

include_flags = '-I' + os.path.dirname(__file__)
cute_include = '-I' + os.path.dirname(__file__) + '/3rd/cutlass/include'

extra_compile_args = {
    'cxx': ['-O2', '-std=c++17', include_flags, cute_include],
    'nvcc': [
        '-arch=sm_90a', '-O2', '-lineinfo', '-Xptxas', '-v', '-std=c++17',
        '--expt-relaxed-constexpr', include_flags, cute_include
    ]
}

extra_link_args = []

cc_files = glob('src/**/*.cc', recursive=True)
cu_files = glob('src/**/*.cu', recursive=True)

sources = cc_files + cu_files
sources = [f for f in sources if not ('test' in f)]

cuda_extension = CUDAExtension(
    name='hpc._C',
    sources=sources,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    py_limited_api=True,
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
