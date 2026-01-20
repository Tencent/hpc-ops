cmake_minimum_required(VERSION 3.18)
project(hpc_ops LANGUAGES CXX CUDA)

enable_language(CUDA)
set(CMAKE_BUILD_TYPE "Release")

find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

file(GLOB_RECURSE SOURCES "src/*/*.cu" "src/*/*.cc")
list(FILTER SOURCES EXCLUDE REGEX ".*test.*")


add_library(_C MODULE ${SOURCES})

set_target_properties(
  _C PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/

  OUTPUT_NAME "_C"
  PREFIX ""
  SUFFIX ".abi3.so"
  CUDA_RUNTIME_LIBRARY "Shared"

  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  SKIP_BUILD_RPATH TRUE

  C_VISIBILITY_PRESET "hidden"
  CXX_VISIBILITY_PRESET "hidden"
  VISIBILITY_INLINES_HIDDEN ON
  CUDA_VISIBILITY_PRESET "hidden"

  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF

  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  CUDA_EXTENSIONS OFF

  CUDA_ARCHITECTURES "90a"
)

target_compile_definitions(
  _C PRIVATE
  Py_LIMITED_API=0x03090000
  _GLIBCXX_USE_CXX11_ABI=1
  HPC_GIT_HASH_STR=${HPC_GIT_HASH_STR}
  HPC_VERSION_STR=${HPC_VERSION_STR}
)

execute_process(
  COMMAND python3 -c "
from torch.utils.cpp_extension import include_paths
print(';'.join(include_paths()), end='')
"
  OUTPUT_VARIABLE TORCH_INCLUDE_PATHS
)

execute_process(
  COMMAND python3 -c "
from torch.utils.cpp_extension import library_paths
print(';'.join(library_paths()), end='')
"
  OUTPUT_VARIABLE TORCH_LIBRARY_PATHS
)


target_include_directories(
  _C PRIVATE
  ./
  3rd/cutlass/include
  ${CUDAToolkit_INCLUDE_DIRS}
  ${TORCH_INCLUDE_PATHS}
)

target_link_directories(
  _C PRIVATE
  ${TORCH_LIBRARY_PATHS}
)

target_link_libraries(
  _C PRIVATE
  cuda
  c10
  torch
  torch_cpu
  cudart
  c10_cuda
  torch_cuda
)

target_compile_options(
  _C PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -Werror=all-warnings
    -lineinfo
    --expt-relaxed-constexpr
    -std=c++17
    -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED
    -DNDEBUG
    -Xptxas=-warn-double-usage,-warn-spills,-Werror,-v
  >
  $<$<COMPILE_LANGUAGE:CXX>:
    -Wno-unused-result
    -Wsign-compare
    -g
    -fwrapv
    -Wall
    -DTORCH_API_INCLUDE_EXTENSION_H
    -DTORCH_EXTENSION_NAME=_C
  >
)
