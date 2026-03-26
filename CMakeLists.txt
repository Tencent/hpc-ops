cmake_minimum_required(VERSION 3.18)
project(hpc_ops LANGUAGES CXX CUDA)

include(ExternalProject)

# Map SM_ARCH number to nvcc architecture string.
# SM90 and SM100 use the 'a' suffix to enable extended ISA (WGMMA, TMA, etc.)
if("${SM_ARCH}" STREQUAL "89")
  set(CUDA_ARCH_STR "89")
elseif("${SM_ARCH}" STREQUAL "90")
  set(CUDA_ARCH_STR "90a")
elseif("${SM_ARCH}" STREQUAL "100")
  set(CUDA_ARCH_STR "100a")
else()
  set(CUDA_ARCH_STR "${SM_ARCH}")
endif()

message(STATUS "Building for SM_ARCH=${SM_ARCH}  CUDA_ARCH=${CUDA_ARCH_STR}")

# ── NVSHMEM (ExternalProject) ─────────────────────────────────────────────────
set(NVSHMEM_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rd/ucl/nvshmem")
set(NVSHMEM_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rd/ucl/nvshmem")

if(NOT EXISTS "${NVSHMEM_INSTALL_DIR}/include")
    file(MAKE_DIRECTORY "${NVSHMEM_INSTALL_DIR}/include")
endif()

find_library(MLX5_LIBRARY
  NAMES mlx5
  HINTS /lib/x86_64-linux-gnu /usr/lib64 /lib64
)

ExternalProject_Add(nvshmem_external
  SOURCE_DIR "${NVSHMEM_SRC_DIR}"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/nvshmem-build"
  CMAKE_ARGS
   -DMLX5_lib=${MLX5_LIBRARY}
   -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH_STR}
   -DNVSHMEM_IBGDA_SUPPORT=ON
   -DCUDA_HOME=/usr/local/cuda
   -DCMAKE_INSTALL_PREFIX=${NVSHMEM_INSTALL_DIR}
   -DNVSHMEM_SHMEM_SUPPORT=OFF
   -DNVSHMEM_MPI_SUPPORT=OFF
   -DNVSHMEM_PMI_SUPPORT=OFF
   -DNVSHMEM_PMIX_SUPPORT=OFF
   -DNVSHMEM_PMI2_SUPPORT=OFF
   -DNVSHMEM_UCX_SUPPORT=OFF
   -DNVSHMEM_USE_NCCL=OFF
   -DNVSHMEM_BUILD_HYDRA_LAUNCHER=OFF
   -DNVSHMEM_BUILD_PYTHON_LIB=OFF
   -DNVSHMEM_BUILD_TXZ_PACKAGE=OFF
   -DNVSHMEM_TIMEOUT_DEVICE_POLLING=OFF
   -DNVSHMEM_USE_GDRCOPY=OFF
   -DNVSHMEM_BUILD_EXAMPLES=OFF
   -DNVSHMEM_BUILD_TESTS=OFF
   -DNVSHMEM_NVTX=OFF
   INSTALL_COMMAND $(MAKE) install
)

add_library(nvshmem_device STATIC IMPORTED GLOBAL)
set_target_properties(
  nvshmem_device PROPERTIES
  IMPORTED_LOCATION "${NVSHMEM_INSTALL_DIR}/lib/libnvshmem_device.a"
  INTERFACE_INCLUDE_DIRECTORIES "${NVSHMEM_INSTALL_DIR}/include"
)
add_dependencies(nvshmem_device nvshmem_external)

# Should Add "${NVSHMEM_INSTALL_DIR}/lib" to "LD_LIBRARY_PATH"
add_library(nvshmem_host SHARED IMPORTED GLOBAL)
set_target_properties(
  nvshmem_host PROPERTIES
  IMPORTED_LOCATION "${NVSHMEM_INSTALL_DIR}/lib/libnvshmem_host.so"
  INTERFACE_INCLUDE_DIRECTORIES "${NVSHMEM_INSTALL_DIR}/include"
)
add_dependencies(nvshmem_host nvshmem_external)

set(UCL_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rd/ucl")
set(UCL_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rd/ucl")

if(NOT EXISTS "${UCL_INSTALL_DIR}/include")
    file(MAKE_DIRECTORY "${UCL_INSTALL_DIR}/include")
endif()

ExternalProject_Add(ucl_external
  SOURCE_DIR "${UCL_SRC_DIR}"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/ucl-build"
  CMAKE_ARGS
   -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH_STR}
   -DCMAKE_INSTALL_PREFIX=${UCL_INSTALL_DIR}
   -DCMAKE_BUILD_TYPE=Release
   -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  BUILD_COMMAND $(MAKE)
  INSTALL_COMMAND $(MAKE) install
  DEPENDS nvshmem_external
)

add_library(ucl STATIC IMPORTED GLOBAL)
set_target_properties(
  ucl PROPERTIES
  IMPORTED_LOCATION ${UCL_INSTALL_DIR}/lib/libucl_hpc.a
  INTERFACE_INCLUDE_DIRECTORIES "${UCL_INSTALL_DIR}/include"
)
add_dependencies(ucl ucl_external)

enable_language(CUDA)
set(CMAKE_BUILD_TYPE "Release")

find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

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

# ── Source file collection ─────────────────────────────────────────────────────
file(GLOB_RECURSE SOURCES "src/*.cc" "src/*.cu")

set(FILTERED_SOURCES "")
foreach(f ${SOURCES})
  if(f MATCHES ".*/sm([0-9]+)/.*")
    if(f MATCHES ".*/sm${SM_ARCH}/.*")
      list(APPEND FILTERED_SOURCES "${f}")
    endif()
  else()
    list(APPEND FILTERED_SOURCES "${f}")
  endif()
endforeach()

list(FILTER FILTERED_SOURCES EXCLUDE REGEX ".*test.*")

set(SOURCES ${FILTERED_SOURCES})

message(STATUS "Source files for SM${SM_ARCH}:")
foreach(f ${SOURCES})
  message(STATUS "  ${f}")
endforeach()
# ──────────────────────────────────────────────────────────────────────────────

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

  CUDA_ARCHITECTURES "${CUDA_ARCH_STR}"
)

target_compile_definitions(
  _C PRIVATE
  Py_LIMITED_API=0x03090000
  _GLIBCXX_USE_CXX11_ABI=1
  HPC_GIT_HASH_STR=\"${HPC_GIT_HASH_STR}\"
  HPC_VERSION_STR=\"${HPC_VERSION_STR}\"
)

target_include_directories(
  _C PRIVATE
  ./
  3rd/cutlass/include
  3rd/ucl/include
  ${NVSHMEM_INSTALL_DIR}/include
  ${CUDAToolkit_INCLUDE_DIRS}
  ${TORCH_INCLUDE_PATHS}
)

target_link_directories(
  _C PRIVATE
  ${TORCH_LIBRARY_PATHS}
)

target_link_libraries(
  _C PRIVATE
  ucl
  nvshmem_device
  nvshmem_host
  cuda
  c10
  torch
  torch_cpu
  cudart
  c10_cuda
  torch_cuda
)

add_dependencies(_C ucl_external nvshmem_external)

# ── Per-architecture CUDA compile flags ───────────────────────────────────────
if("${SM_ARCH}" STREQUAL "90")
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
elseif("${SM_ARCH}" STREQUAL "100")
  target_compile_options(
    _C PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      -Werror=all-warnings
      -lineinfo
      --expt-relaxed-constexpr
      -std=c++17
      -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED
      -DNDEBUG
      -Xptxas=-warn-double-usage,-Werror,-v
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
elseif("${SM_ARCH}" STREQUAL "89")
  target_compile_options(
    _C PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      -Werror=all-warnings
      -lineinfo
      --expt-relaxed-constexpr
      -std=c++17
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
else()
  target_compile_options(
    _C PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      -Werror=all-warnings
      -lineinfo
      --expt-relaxed-constexpr
      -std=c++17
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