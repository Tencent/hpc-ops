cmake_minimum_required(VERSION 3.18)
project(hpc_ops LANGUAGES CXX CUDA)

enable_language(CUDA)
set(CMAKE_BUILD_TYPE "Release")

find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

set(HPC_OPS_CUDA_ARCHITECTURES
    "90a"
    CACHE STRING "CUDA architectures used to build hpc-ops kernels")

file(GLOB_RECURSE SOURCES "src/*/*.cu" "src/*/*.cc")
file(GLOB CP_ASYNC_SOURCES "src/fuse_moe/cp_async/*.cu" "src/group_gemm/cp_async/*.cu"
                           "src/group_gemm/cp_async/*.cc")
list(APPEND SOURCES ${CP_ASYNC_SOURCES})
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

  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF

  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  CUDA_EXTENSIONS OFF

  CUDA_ARCHITECTURES "${HPC_OPS_CUDA_ARCHITECTURES}"
)

function(query_python output_variable description script)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "${script}"
    RESULT_VARIABLE PYTHON_QUERY_RESULT
    OUTPUT_VARIABLE PYTHON_QUERY_OUTPUT
    ERROR_VARIABLE PYTHON_QUERY_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
  )
  if(NOT PYTHON_QUERY_RESULT EQUAL 0)
    message(FATAL_ERROR
            "Failed to query ${description} with ${Python3_EXECUTABLE}.\n${PYTHON_QUERY_ERROR}")
  endif()
  if("${PYTHON_QUERY_OUTPUT}" STREQUAL "")
    message(FATAL_ERROR "Python query for ${description} returned an empty result.")
  endif()
  set(${output_variable}
      "${PYTHON_QUERY_OUTPUT}"
      PARENT_SCOPE)
endfunction()

query_python(
  TORCH_VERSION
  "PyTorch version"
  "
import torch
print(torch.__version__, end='')
")

query_python(
  TORCH_CUDA_VERSION
  "PyTorch CUDA version"
  "
import torch
print(torch.version.cuda or 'none', end='')
")

query_python(
  TORCH_CXX11_ABI
  "PyTorch CXX11 ABI"
  "
import torch
print(int(getattr(torch._C, '_GLIBCXX_USE_CXX11_ABI', True)), end='')
")

set(HPC_OPS_USE_CXX11_ABI
    "${TORCH_CXX11_ABI}"
    CACHE STRING "Value for _GLIBCXX_USE_CXX11_ABI; defaults to the active PyTorch ABI")

query_python(
  TORCH_INCLUDE_PATHS
  "PyTorch include paths"
  "
from torch.utils.cpp_extension import include_paths
print(chr(59).join(include_paths()), end='')
")

query_python(
  TORCH_LIBRARY_PATHS
  "PyTorch library paths"
  "
from torch.utils.cpp_extension import library_paths
print(chr(59).join(library_paths()), end='')
")

message(STATUS "HPC-Ops build Python: ${Python3_EXECUTABLE}")
message(STATUS "HPC-Ops build PyTorch: ${TORCH_VERSION} (CUDA ${TORCH_CUDA_VERSION})")
message(STATUS "HPC-Ops PyTorch CXX11 ABI: ${HPC_OPS_USE_CXX11_ABI}")
message(STATUS "HPC-Ops CUDA architectures: ${HPC_OPS_CUDA_ARCHITECTURES}")
message(STATUS "HPC-Ops C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "HPC-Ops CUDA compiler: ${CMAKE_CUDA_COMPILER_VERSION}")

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
  message(
    STATUS
      "GCC 13+ detected. If compilation fails inside PyTorch headers, verify that the PyTorch, CUDA, and host compiler versions are supported together."
  )
endif()

target_compile_definitions(
  _C PRIVATE
  Py_LIMITED_API=0x03090000
  _GLIBCXX_USE_CXX11_ABI=${HPC_OPS_USE_CXX11_ABI}
  HPC_GIT_HASH_STR=${HPC_GIT_HASH_STR}
  HPC_VERSION_STR=${HPC_VERSION_STR}
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
    -std=c++20
    -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED
    -DNDEBUG
    -Xptxas=-warn-double-usage,-v
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
