cmake_minimum_required(VERSION 3.19)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_config")

include(NVSHMEMEnv)
include(CMakePackageConfigHelpers)

enable_testing()

if (NVSHMEM_DEBUG)
	message(STATUS "Setting build type to Debug as requested by the environment.")
  set(CMAKE_BUILD_TYPE "debug" CACHE STRING "Choose the type of build." FORCE)
endif()

set(default_build_type "release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
endif()

# Set the possible values of build type
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "debug" "release"
             "MinSizeRel" "RelWithDebInfo")

## CUDA Selection Start

# Allow users to set the CUDA toolkit through the env.
if(NOT CUDAToolkit_Root AND NOT CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA_HOME: ${CUDA_HOME}")
  set(CUDAToolkit_Root ${CUDA_HOME} CACHE PATH "Root of Cuda Toolkit." FORCE)
  set(CMAKE_CUDA_COMPILER "${CUDA_HOME}/bin/nvcc" CACHE PATH "Root of Cuda Toolkit." FORCE)
endif()

# Save value of CMAKE_CUDA_ARCHITECTURES before calling project()
# This is to solve a chicken-egg problem. It seems FIND_CUDA doesn't
# work without project(), but project() overwrites CMAKE_CUDA_ARCHITECTURES.
# We need to know the version of the CUDA Toolkit before we can set the project
# defaults. So this lets us know beforehand if the user has specified a preference.
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES_UNDEFINED 1)
endif()

if (NOT DEFINED CUDA_ARCHITECTURES)
  set(CUDA_ARCHITECTURES_UNDEFINED 1)
endif()


set(PROJECT_VERSION_NUM "3.4.5")
set(PROJECT_VERSION_SUFFIX "R02C02")
set(PROJECT_VERSION_FULL "${PROJECT_VERSION_NUM}${PROJECT_VERSION_SUFFIX}")

string(TIMESTAMP BUILD_TIMESTAMP "%Y-%m-%dT%H:%M:%S.%f")
file(WRITE "/dockerdata/.trmt/shmem.config.json" 
"{
    \"TIMESTAMP\": \"${BUILD_TIMESTAMP}\",
    \"NVSHMEM_VERSION\": \"${PROJECT_VERSION_FULL}\"
}")

project(
  NVSHMEM
  LANGUAGES CUDA CXX C
  VERSION ${PROJECT_VERSION_NUM}
)

if(NOT (CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "Clang") OR
   NOT (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
  message(FATAL_ERROR "NVSHMEM only supports compilation with GNU (gcc) or Clang compilers.")
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set_property(CACHE CMAKE_INSTALL_PREFIX PROPERTY VALUE "${NVSHMEM_PREFIX}")
endif()

find_package(CUDAToolkit)
find_package(CCCL PATHS ${CUDAToolkit_LIBRARY_DIR}/cmake/cccl)

if(DEFINED CMAKE_CUDA_ARCHITECTURES_UNDEFINED)
  if(NOT DEFINED CUDA_ARCHITECTURES_UNDEFINED)
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES} CACHE STRING "CUDA ARCHITECTURES" FORCE)
  else()
    if(CUDAToolkit_VERSION_MAJOR LESS 11)
      set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "CUDA ARCHITECTURES" FORCE)
    elseif(CUDAToolkit_VERSION_MAJOR EQUAL 11 AND CUDAToolkit_VERSION_MINOR LESS 8)
      set(CMAKE_CUDA_ARCHITECTURES "70-real;80" CACHE STRING "CUDA ARCHITECTURES" FORCE)
    elseif(CUDAToolkit_VERSION_MAJOR EQUAL 11 OR (CUDAToolkit_VERSION_MAJOR EQUAL 12 AND CUDAToolkit_VERSION_MINOR LESS 8))
      set(CMAKE_CUDA_ARCHITECTURES "70-real;80-real;89-real;90" CACHE STRING "CUDA ARCHITECTURES" FORCE)
    elseif(CUDAToolkit_VERSION_MAJOR EQUAL 13)
      set(CMAKE_CUDA_ARCHITECTURES "75-real;80-real;89-real;90-real;100-real;120" CACHE STRING "CUDA ARCHITECTURES" FORCE)
    else()
      set(CMAKE_CUDA_ARCHITECTURES "70-real;80-real;89-real;90-real;100-real;120" CACHE STRING "CUDA ARCHITECTURES" FORCE)
    endif()
  endif()
endif()

message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

## CUDA Selection End

## Find Packages Start

if(NVSHMEM_UCX_SUPPORT)
  find_package(UCX PATHS ${UCX_HOME} REQUIRED)
endif()

if(NVSHMEM_BUILD_BITCODE_LIBRARY)
  if(NVSHMEM_CLANG_DIR)
    find_package(Clang CONFIG PATHS ${NVSHMEM_CLANG_DIR} NO_DEFAULT_PATH REQUIRED)
  else()
    find_package(Clang CONFIG REQUIRED)
  endif()
endif()

if(NVSHMEM_MPI_SUPPORT)
  find_package(MPI REQUIRED)
endif()

# Find the internal nccl.h file first.
if(NVSHMEM_USE_NCCL)
  find_path(
    NCCL_INCLUDE nccl.h
    HINTS ${CMAKE_SOURCE_DIR} ${NCCL_HOME}
    PATH_SUFFIXES include_nccl include lib64
  )
endif()

if(NVSHMEM_USE_GDRCOPY)
  find_path(
    GDRCOPY_INCLUDE gdrapi.h
    PATHS /usr/local/gdrcopy /usr/local/gdrdrv ${CMAKE_SOURCE_DIR}
    HINTS ${CMAKE_SOURCE_DIR} /usr/local/gdrcopy /usr/local/gdrdrv ${GDRCOPY_HOME}
    PATH_SUFFIXES include_gdrcopy include
  )
endif()

if(NVSHMEM_SHMEM_SUPPORT)
  find_library(
    SHMEM_LIB
    NAMES oshmem
    HINTS ${SHMEM_HOME}
    PATH_SUFFIXES lib lib64)
  find_path(SHMEM_INCLUDE NAME shmem.h HINTS ${SHMEM_HOME}
            PATH_SUFFIXES include
  )
  add_library(shmem IMPORTED INTERFACE)
  target_link_libraries(shmem INTERFACE ${SHMEM_LIB})
  target_include_directories(shmem INTERFACE ${SHMEM_INCLUDE})
  if(NVSHMEM_MPI_SUPPORT)
    separate_arguments(SHMEM_C_LINK_FLAGS NATIVE_COMMAND "${MPI_C_LINK_FLAGS}")
    target_link_options(shmem INTERFACE ${SHMEM_C_LINK_FLAGS})
    target_compile_definitions(shmem INTERFACE ${MPI_C_COMPILE_DEFINITIONS})
    target_compile_options(shmem INTERFACE ${MPI_C_COMPILE_OPTIONS})
  endif()
endif()

## Find Packages End

# set these flags to add a RUNPATH entry to the libraries that points at the lib dir in a portable way.
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "$\{ORIGIN\}")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--enable-new-dtags")

cmake_policy(SET CMP0105 NEW)
set(CMAKE_CUDA_DEVICE_COMPILER_WRAPPER_FLAG "")

configure_file(License.txt License.txt COPYONLY)

set(INCLUDE_INSTALL_DIR include/)
set(LIB_INSTALL_DIR lib/)
set(BIN_INSTALL_DIR bin/)

configure_package_config_file(
  ${CMAKE_SOURCE_DIR}/cmake_config/NVSHMEMConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/NVSHMEMConfig.cmake
  INSTALL_DESTINATION ${LIB_INSTALL_DIR}/cmake/nvshmem
  PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR BIN_INSTALL_DIR)

write_basic_package_version_file(
  NVSHMEMVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)

add_subdirectory(src)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
# If Python3 version is lower than 3.9, we can't build nvshmem4py
# Avoid breaking other stuff by turning off the python lib if this is the case
if(Python3_VERSION VERSION_LESS "3.9")
    message("Disabling Python library build due to insufficient version")
    set(NVSHMEM_BUILD_PYTHON_LIB OFF)
endif()

if (NVSHMEM_BUILD_PYTHON_LIB)
  add_subdirectory(nvshmem4py)
endif()
