# Copyright (C) 2026 Tencent.
#
# One module per architecture. See CLAUDE.md for layout and source selection.

cmake_minimum_required(VERSION 3.26)
project(hpc_ops LANGUAGES CXX CUDA)

enable_language(CUDA)
set(CMAKE_BUILD_TYPE "Release")

find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# ============================================================
#    Architecture selection
# ============================================================

# Architectures the build supports. The same list is in the Makefile and
# setup.py; adding one also needs an HPC_ARCH_EVAL_<arch> block in
# src/utils/utils.h.
set(HPC_KNOWN_ARCHS 90 100 103)

# Architectures whose gencode needs the "a" suffix for wgmma/TMA instructions.
set(HPC_ARCH_SPECIFIC_ARCHS 90 100 103)

# Source fallback: an architecture with no sm<arch>/ of its own inherits the
# listed one's sources, recompiled with its own gencode. Only sm103 -> sm100,
# which share an instruction set; a link is a claim that has to be tested on
# that hardware.
set(HPC_ARCH_SOURCE_FALLBACK_103 100)

if(NOT HPC_TARGET_ARCH IN_LIST HPC_KNOWN_ARCHS)
  message(FATAL_ERROR
          "HPC_TARGET_ARCH is '${HPC_TARGET_ARCH}', expected one of ${HPC_KNOWN_ARCHS}. "
          "One configure builds one architecture; build through the Makefile "
          "(make / make sm90 / SM_ARCH=90,100 make).")
endif()

if(HPC_TARGET_ARCH IN_LIST HPC_ARCH_SPECIFIC_ARCHS)
  set(HPC_GENCODE "${HPC_TARGET_ARCH}a")
else()
  set(HPC_GENCODE "${HPC_TARGET_ARCH}")
endif()

set(HPC_SOURCE_ARCH "${HPC_ARCH_SOURCE_FALLBACK_${HPC_TARGET_ARCH}}")

message(STATUS "hpc-ops: building sm${HPC_TARGET_ARCH} -> gencode sm_${HPC_GENCODE}")

# ============================================================
#    Source selection
# ============================================================
#
# Compile a source iff its path belongs to this architecture:
#   1. no sm<N>/          — shared, always compiled
#   2. sm<TARGET>/        — this architecture
#   3. sm<FALLBACK>/      — only where sm<TARGET>/ does not exist beside it
#   4. other sm<N>/       — skipped

file(GLOB_RECURSE HPC_CANDIDATE_SOURCES RELATIVE "${CMAKE_SOURCE_DIR}"
     "${CMAKE_SOURCE_DIR}/src/*.cu" "${CMAKE_SOURCE_DIR}/src/*.cc")
list(FILTER HPC_CANDIDATE_SOURCES EXCLUDE REGEX ".*test.*")

set(HPC_SOURCES "")
foreach(rel IN LISTS HPC_CANDIDATE_SOURCES)
  get_filename_component(dir "${rel}" DIRECTORY)

  if(NOT dir MATCHES "(^|/)sm([0-9]+)(/|$)")
    list(APPEND HPC_SOURCES "${CMAKE_SOURCE_DIR}/${rel}")  # rule 1
    continue()
  endif()

  set(src_arch "${CMAKE_MATCH_2}")

  if(src_arch STREQUAL HPC_TARGET_ARCH)
    list(APPEND HPC_SOURCES "${CMAKE_SOURCE_DIR}/${rel}")  # rule 2
  elseif(src_arch STREQUAL HPC_SOURCE_ARCH)
    # Rule 3: the target's own directory, if it exists, wins over the fallback.
    string(REGEX REPLACE "(^|/)sm${src_arch}(/|$)" "\\1sm${HPC_TARGET_ARCH}\\2" own "${dir}")
    if(NOT IS_DIRECTORY "${CMAKE_SOURCE_DIR}/${own}")
      list(APPEND HPC_SOURCES "${CMAKE_SOURCE_DIR}/${rel}")
    endif()
  endif()
endforeach()

# ============================================================
#    Compile settings
# ============================================================

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "
from torch.utils.cpp_extension import include_paths
print(';'.join(include_paths()), end='')
"
  OUTPUT_VARIABLE TORCH_INCLUDE_PATHS
)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "
from torch.utils.cpp_extension import library_paths
print(';'.join(library_paths()), end='')
"
  OUTPUT_VARIABLE TORCH_LIBRARY_PATHS
)

set(HPC_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/3rd/cutlass/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TORCH_INCLUDE_PATHS})

# ============================================================
#    The _C_sm<arch> python module
# ============================================================

set(HPC_MODULE_NAME "_C_sm${HPC_TARGET_ARCH}")

add_library(${HPC_MODULE_NAME} MODULE ${HPC_SOURCES})

set_target_properties(
  ${HPC_MODULE_NAME} PROPERTIES
  CUDA_RUNTIME_LIBRARY "Shared"

  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  SKIP_BUILD_RPATH TRUE

  POSITION_INDEPENDENT_CODE ON

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

  CUDA_ARCHITECTURES "${HPC_GENCODE}"

  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/
  OUTPUT_NAME "${HPC_MODULE_NAME}"
  PREFIX ""
  SUFFIX ".abi3.so"
)

target_include_directories(${HPC_MODULE_NAME} PRIVATE ${HPC_INCLUDE_DIRS})

target_compile_definitions(
  ${HPC_MODULE_NAME} PRIVATE
  Py_LIMITED_API=0x03090000
  _GLIBCXX_USE_CXX11_ABI=1
  HPC_GIT_HASH_STR=${HPC_GIT_HASH_STR}
  HPC_VERSION_STR=${HPC_VERSION_STR}

  # Dispatch key: only HPC_ARCH_EVAL_${HPC_TARGET_ARCH} expands to a call.
  HPC_TARGET_ARCH=${HPC_TARGET_ARCH}
)

target_compile_options(
  ${HPC_MODULE_NAME} PRIVATE
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
    -DTORCH_EXTENSION_NAME=${HPC_MODULE_NAME}
  >
)

target_link_directories(
  ${HPC_MODULE_NAME} PRIVATE
  ${TORCH_LIBRARY_PATHS}
)

# A MODULE library tolerates undefined symbols by default, so an HPC_ARCH_DISPATCH
# pair for an architecture with no sm<arch>/ would link and then fail at import
# with nothing but "could not load this library". Fail at the link instead, where
# the error names the symbol and the entry.cc line that referenced it.
target_link_options(${HPC_MODULE_NAME} PRIVATE -Wl,--no-undefined)

target_link_libraries(
  ${HPC_MODULE_NAME} PRIVATE
  cuda
  c10
  torch
  torch_cpu
  cudart
  c10_cuda
  torch_cuda
)
