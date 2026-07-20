cmake_minimum_required(VERSION 3.18)

# ROCm/HIP support is opt-in via -DUSE_ROCM=ON (or the USE_ROCM env var). When
# off, the build is byte-for-byte the original NVIDIA CUDA path. When on, only
# the already-ported operators under src/amd/ are compiled for AMD GPUs; the
# CUDA-only kernels (attention/moe/gemm/... which depend on CUTLASS/CuTe/Hopper
# TMA) are excluded because they do not build on AMD.
option(USE_ROCM "Build the ROCm/HIP backend instead of CUDA" OFF)
if(NOT USE_ROCM AND DEFINED ENV{USE_ROCM})
  set(USE_ROCM $ENV{USE_ROCM})
endif()

if(USE_ROCM)
  # Default to gfx950 (MI350X), but let PYTORCH_ROCM_ARCH override it. Must be
  # set before enable_language(HIP) so the HIP compiler picks up the target.
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    if(DEFINED ENV{PYTORCH_ROCM_ARCH})
      set(CMAKE_HIP_ARCHITECTURES $ENV{PYTORCH_ROCM_ARCH})
    else()
      set(CMAKE_HIP_ARCHITECTURES gfx950)
    endif()
  endif()
  project(hpc_ops LANGUAGES CXX HIP)
  enable_language(HIP)
  find_package(hip REQUIRED)
else()
  project(hpc_ops LANGUAGES CXX CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

set(CMAKE_BUILD_TYPE "Release")

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

if(USE_ROCM)
  # ROCm: explicit allowlist of ported sources only. Do NOT glob src/*, that
  # would pull in the CUDA-only kernels. Keep paths free of the substring
  # "test" (the CUDA glob below filters it, and we mirror that convention).
  set(SOURCES
      ${CMAKE_SOURCE_DIR}/src/C/C.cc
      ${CMAKE_SOURCE_DIR}/src/C/version.cc
      ${CMAKE_SOURCE_DIR}/src/amd/C/built_json.cc
      ${CMAKE_SOURCE_DIR}/src/amd/normalization/entry.cc
      ${CMAKE_SOURCE_DIR}/src/amd/normalization/fused_rmsnorm_with_scale.cu)
  # Treat .cu files as HIP so hipcc compiles them.
  set_source_files_properties(
    ${CMAKE_SOURCE_DIR}/src/amd/normalization/fused_rmsnorm_with_scale.cu
    PROPERTIES LANGUAGE HIP)
else()
  file(GLOB_RECURSE SOURCES "src/*/*.cu" "src/*/*.cc")
  file(GLOB CP_ASYNC_SOURCES "src/fuse_moe/cp_async/*.cu" "src/group_gemm/cp_async/*.cu"
                             "src/group_gemm/cp_async/*.cc")
  list(APPEND SOURCES ${CP_ASYNC_SOURCES})
  list(FILTER SOURCES EXCLUDE REGEX ".*test.*")
  # Exclude the ROCm-only tree from the CUDA build.
  list(FILTER SOURCES EXCLUDE REGEX ".*/src/amd/.*")
endif()


add_library(_C MODULE ${SOURCES})

if(USE_ROCM)
  set_target_properties(
    _C PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/

    OUTPUT_NAME "_C"
    PREFIX ""
    SUFFIX ".abi3.so"

    SKIP_BUILD_RPATH TRUE

    C_VISIBILITY_PRESET "hidden"
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    HIP_VISIBILITY_PRESET "hidden"

    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF

    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    HIP_EXTENSIONS OFF)
else()
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

    CUDA_ARCHITECTURES "90a")
endif()

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


if(USE_ROCM)
  target_include_directories(
    _C PRIVATE
    ./
    ${TORCH_INCLUDE_PATHS}
    $ENV{ROCM_PATH}/include
    /opt/rocm/include
  )
else()
  target_include_directories(
    _C PRIVATE
    ./
    3rd/cutlass/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TORCH_INCLUDE_PATHS}
  )
endif()

target_link_directories(
  _C PRIVATE
  ${TORCH_LIBRARY_PATHS}
)

if(USE_ROCM)
  target_link_libraries(
    _C PRIVATE
    hip::host
    c10
    torch
    torch_cpu
    c10_hip
    torch_hip
  )
else()
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
endif()

if(USE_ROCM)
  # ROCm PyTorch requires these macros so the CUDA-named ATen/torch headers
  # resolve to their HIP backing (same set torch's cpp_extension injects).
  target_compile_definitions(
    _C PRIVATE
    __HIP_PLATFORM_AMD__=1
    USE_ROCM=1
    HIPBLAS_V2
  )
  target_compile_options(
    _C PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:
      -std=c++17
      -DNDEBUG
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
else()
  target_compile_options(
    _C PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      -Werror=all-warnings
      -lineinfo
      --expt-relaxed-constexpr
      -std=c++17
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
endif()
