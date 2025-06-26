from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
import os
from glob import glob

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

mm_files = ['src/_C.cc']
cc_files = glob('src/*/*.cc')
cu_files = glob('src/*/*.cu')

sources = mm_files + cc_files + cu_files
sources = [f for f in sources if not f.endswith('test.cc')]

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
