cmake_minimum_required(VERSION 3.27 FATAL_ERROR)

# Detect CUDA toolkit: tries host installation first, then falls back to
# pip-installed packages (env WITH_PIP_CUDA_TOOLCHAIN or auto-detect).
# Must be included before project() so CMAKE_CUDA_COMPILER is set.
include(${CMAKE_CURRENT_LIST_DIR}/cmake/FindPipCUDAToolkit.cmake)

project(hpc_ops LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_RUNTIME_LIBRARY None)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_ARCHITECTURES "90a")

find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
link_directories(${CUDAToolkit_LIBRARY_DIR} ${CUDAToolkit_LIBRARY_DIR}/stubs)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
   set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_FOUND}")
   set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_FOUND}")
   set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_FOUND}")
endif()

find_package(
  Python
  COMPONENTS Interpreter
  REQUIRED
)
find_package(tvm_ffi CONFIG REQUIRED)

if(NOT "${SKBUILD_SABI_VERSION}" STREQUAL "")
  set(USE_SABI USE_SABI ${SKBUILD_SABI_VERSION})
endif()

include_directories(
  ./
  src/utils/include
  3rd/cutlass/include
)

# Collect all CUDA source files (kernels)
file(GLOB_RECURSE CUDA_SOURCES "src/*/*.cu")
list(FILTER CUDA_SOURCES EXCLUDE REGEX ".*test.*")
# Exclude built_json.cu as it's compiled separately
list(FILTER CUDA_SOURCES EXCLUDE REGEX ".*/C/built_json\\.cu$")

# Collect all CC source files (entry points)
file(GLOB_RECURSE CC_SOURCES "src/*/*.cc")
list(FILTER CC_SOURCES EXCLUDE REGEX ".*test.*")
# Exclude C.cc placeholder
list(FILTER CC_SOURCES EXCLUDE REGEX ".*/C/C\\.cc$")

# Build all kernels as a single shared library
add_library(_C SHARED ${CUDA_SOURCES} ${CC_SOURCES} src/C/built_json.cu)
target_link_libraries(_C PRIVATE tvm_ffi::shared cuda cudart)
set_target_properties(
  _C PROPERTIES
  PREFIX ""
  SUFFIX ".so"
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/

  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  SKIP_BUILD_RPATH TRUE

  C_VISIBILITY_PRESET "hidden"
  CXX_VISIBILITY_PRESET "hidden"
  VISIBILITY_INLINES_HIDDEN ON

  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF

  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  CUDA_EXTENSIONS OFF

  CUDA_ARCHITECTURES "90a"
)

if(NOT DEFINED HPC_GIT_HASH_STR OR HPC_GIT_HASH_STR STREQUAL "")
  execute_process(
    COMMAND git rev-parse --short=7 HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE HPC_GIT_HASH_STR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(NOT HPC_GIT_HASH_STR)
    set(HPC_GIT_HASH_STR "unknown")
  endif()
endif()

if(NOT DEFINED HPC_VERSION_STR OR HPC_VERSION_STR STREQUAL "")
  set(HPC_VERSION_STR "0.0.1-dev")
endif()

target_compile_definitions(
  _C PRIVATE
  _GLIBCXX_USE_CXX11_ABI=1
  HPC_GIT_HASH_STR="${HPC_GIT_HASH_STR}"
  HPC_VERSION_STR="${HPC_VERSION_STR}"
)

target_compile_options(
  _C PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -Werror=all-warnings
    -Wno-error=deprecated-declarations
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
  >
)

install(TARGETS _C LIBRARY DESTINATION hpc/ops)
