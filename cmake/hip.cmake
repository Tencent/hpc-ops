# Copyright (C) 2026 Tencent.
# AMD sources are selected explicitly: CUDA shared sources contain NVIDIA PTX.
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
          "from torch.utils.cpp_extension import include_paths; print(';'.join(include_paths()))"
  OUTPUT_VARIABLE TORCH_INCLUDE_PATHS OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
          "from torch.utils.cpp_extension import library_paths; print(';'.join(library_paths(device_type='cuda')))"
  OUTPUT_VARIABLE TORCH_LIBRARY_PATHS OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
          "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
  OUTPUT_VARIABLE TORCH_CXX11_ABI OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)

add_library(_C_hip MODULE
  src/C/C.cc
  src/C/version.cc
  src/C/amd/built_json.cc
  src/C/amd/unsupported.cc
  src/rope/amd/entry.cc
  src/rope/amd/rope.hip
  src/fuse_moe/amd/entry.cc
  src/fuse_moe/amd/reduce.hip
  src/iHC/amd/entry.cc
  src/iHC/amd/iHC.hip)
set_target_properties(_C_hip PROPERTIES
  CXX_STANDARD 20 HIP_STANDARD 20
  CXX_STANDARD_REQUIRED ON HIP_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON PREFIX "" SUFFIX ".abi3.so"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "${TORCH_LIBRARY_PATHS}"
  INSTALL_RPATH "$ORIGIN/../torch/lib")
target_include_directories(_C_hip PRIVATE
  ${CMAKE_SOURCE_DIR} ${Python3_INCLUDE_DIRS} ${TORCH_INCLUDE_PATHS}
  ${CMAKE_HIP_COMPILER_ROCM_ROOT}/include)
string(REPLACE ";" "," HPC_HIP_ARCH_CSV "${CMAKE_HIP_ARCHITECTURES}")
target_compile_definitions(_C_hip PRIVATE
  __HIP_PLATFORM_AMD__ USE_ROCM
  _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}
  HPC_VERSION_STR=${HPC_VERSION_STR} HPC_GIT_HASH_STR=${HPC_GIT_HASH_STR}
  HPC_HIP_ARCH_STR="${HPC_HIP_ARCH_CSV}")
# Preserve separate FP32 multiply/add rounding, including iHC post's BF16 cast.
target_compile_options(_C_hip PRIVATE
  -O3 $<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>)
target_link_directories(_C_hip PRIVATE ${TORCH_LIBRARY_PATHS}
  ${CMAKE_HIP_COMPILER_ROCM_ROOT}/lib)
target_link_libraries(_C_hip PRIVATE torch torch_cpu torch_hip c10 c10_hip amdhip64)
target_link_options(_C_hip PRIVATE -Wl,--no-undefined)
