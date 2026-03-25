cmake_minimum_required(VERSION 3.19)

project(UCL VERSION 1.0.0 LANGUAGES CUDA CXX C)

# Set default build type
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
endif()

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA related variables
if(NOT CUDAToolkit_Root AND NOT CMAKE_CUDA_COMPILER)
  if(DEFINED ENV{CUDA_HOME})
    set(CUDA_HOME $ENV{CUDA_HOME})
  else()
    set(CUDA_HOME "/usr/local/cuda")
  endif()
  set(CUDAToolkit_Root ${CUDA_HOME} CACHE PATH "Root of Cuda Toolkit." FORCE)
  set(CMAKE_CUDA_COMPILER "${CUDA_HOME}/bin/nvcc" CACHE PATH "CUDA Compiler." FORCE)
endif()

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Set install prefix (default to current source directory)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "Install prefix" FORCE)
endif()

# Set CUDA architectures (default to 90a, can be overridden via -DCMAKE_CUDA_ARCHITECTURES)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "90a" CACHE STRING "CUDA architectures" FORCE)
endif()

message(STATUS "UCL: Building with CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

#==========================================
# NVSHMEM Dependency (Pre-installed)
#==========================================

# NVSHMEM must be pre-installed in nvshmem subdirectory
set(NVSHMEM_HOME "${CMAKE_CURRENT_SOURCE_DIR}/nvshmem")
set(NVSHMEM_INCLUDE_DIR "${NVSHMEM_HOME}/include")
set(NVSHMEM_LIB_DIR "${NVSHMEM_HOME}/lib")

# Verify NVSHMEM installation
if(NOT EXISTS "${NVSHMEM_LIB_DIR}/libnvshmem_device.a" OR 
   NOT EXISTS "${NVSHMEM_LIB_DIR}/libnvshmem_host.so")
    message(FATAL_ERROR "NVSHMEM not found in ${NVSHMEM_HOME}. Please build NVSHMEM first.")
endif()

message(STATUS "Using pre-installed NVSHMEM in ${NVSHMEM_HOME}")

# Create NVSHMEM imported targets
add_library(nvshmem::nvshmem_host SHARED IMPORTED GLOBAL)
set_target_properties(nvshmem::nvshmem_host PROPERTIES
    IMPORTED_LOCATION "${NVSHMEM_LIB_DIR}/libnvshmem_host.so"
    INTERFACE_INCLUDE_DIRECTORIES "${NVSHMEM_INCLUDE_DIR}"
)

add_library(nvshmem::nvshmem_device STATIC IMPORTED GLOBAL)
set_target_properties(nvshmem::nvshmem_device PROPERTIES
    IMPORTED_LOCATION "${NVSHMEM_LIB_DIR}/libnvshmem_device.a"
    INTERFACE_INCLUDE_DIRECTORIES "${NVSHMEM_INCLUDE_DIR}"
)

#==========================================
# UCL Library Build
#==========================================

# UCL source files
set(UCL_SOURCES
    src/host/shmem_host.cpp
    src/device/shmem_device.cu
)

# Create UCL static library
add_library(ucl STATIC ${UCL_SOURCES})

# Set CUDA properties for UCL
set_target_properties(ucl PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "ucl_hpc"
)

# Set UCL include directories
target_include_directories(ucl
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
    $<BUILD_INTERFACE:${NVSHMEM_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Link dependencies
target_link_libraries(ucl
  PUBLIC
    CUDA::cudart
    nvshmem::nvshmem_host
    nvshmem::nvshmem_device
)

#==========================================
# Installation Rules
#==========================================

# Install UCL headers (preserve directory structure)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/include/ucl/
  DESTINATION include/ucl
  FILES_MATCHING 
  PATTERN "*.h"
  PATTERN "*.cuh"
  PATTERN "*.hpp"
)

# Install UCL library
install(TARGETS ucl
  EXPORT UCLTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

# Create UCL export targets
install(EXPORT UCLTargets
  FILE UCLTargets.cmake
  NAMESPACE ucl::
  DESTINATION lib/cmake/ucl
)

# Create version file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/UCLConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

# Set install directories for config file
set(INCLUDE_INSTALL_DIR include)
set(LIB_INSTALL_DIR lib)
set(NVSHMEM_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/nvshmem")

# Create config file
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/UCLConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/UCLConfig.cmake"
  INSTALL_DESTINATION lib/cmake/ucl
  PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR NVSHMEM_INSTALL_DIR
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# Install config files
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/UCLConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/UCLConfigVersion.cmake"
  DESTINATION lib/cmake/ucl
)

message(STATUS "UCL configuration complete")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  NVSHMEM directory: ${NVSHMEM_HOME}")
