cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(hpc_ext LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 90a)
option(ATen_CUDA_ARCH_LIST "Override arch list" ON)

set(TORCH_CUDA_ARCH_LIST "9.0a" CACHE STRING "Forced CUDA architectures")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB mm_files "src/_C.cc")
file(GLOB_RECURSE cc_files "src/*/*.cc")
file(GLOB_RECURSE cu_files "src/*/*.cu")

set(all_sources ${mm_files} ${cc_files} ${cu_files})
list(FILTER all_sources EXCLUDE REGEX ".*test\\.cc$")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 --expt-relaxed-constexpr")

add_library(_C SHARED ${all_sources})
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES})

set_target_properties(_C PROPERTIES
    CXX_EXTENSIONS OFF
    CUDA_EXTENSIONS OFF
    CUDA_ARCHITECTURES "90a"
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "_C"
    PREFIX ""
    SUFFIX ".abi3.so"
)

message(STATUS "Final CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Torch CUDA arch list: ${TORCH_CUDA_ARCH_LIST}")

install(TARGETS _C
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/hpc
)
