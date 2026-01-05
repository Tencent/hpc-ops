cmake_minimum_required(VERSION 3.18)
project(hpc_ops LANGUAGES CXX CUDA)

include(ExternalProject)

set(NVSHMEM_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rd/nvshmem")
set(NVSHMEM_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/nvshmem")

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
   -DCMAKE_CUDA_ARCHITECTURES=90a
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
   INSTALL_COMMAND $(MAKE) install
)

add_library(nvshmem STATIC IMPORTED GLOBAL)
set_target_properties(
  nvshmem PROPERTIES
  IMPORTED_LOCATION ${NVSHMEM_INSTALL_DIR}/lib/libnvshmem.a
  INTERFACE_INCLUDE_DIRECTORIES ${NVSHMEM_INSTALL_DIR}/include
)

add_dependencies(nvshmem nvshmem_external)

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

target_include_directories(
  _C PRIVATE
  ./
  3rd/cutlass/include
  ${Python3_SITELIB}/torch/include
  ${Python3_SITELIB}/torch/include/torch/csrc/api/include
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_directories(
  _C PRIVATE
  ${Python3_SITELIB}/torch/lib
)

target_link_libraries(
  _C PRIVATE
  nvshmem
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
