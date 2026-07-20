// Copyright (C) 2026 Tencent.
// ROCm/HIP build-info reporter. Mirrors src/C/built_json.cu but reports the HIP
// toolchain instead of nvcc/cutlass/cub (which do not exist in a ROCm build).

#include <hip/hip_version.h>
#include <torch/library.h>

#include <sstream>
#include <string>

#ifndef HPC_VERSION_STR
#define HPC_VERSION_STR "unknown"
#endif

#ifndef HPC_GIT_HASH_STR
#define HPC_GIT_HASH_STR "unknown"
#endif

static const std::string built_json() {
  std::ostringstream oss;

  // NOLINTBEGIN clang-format off
  oss << "{" << "\n";
  oss << " \"version\": " << "\"" << HPC_VERSION_STR << "\",\n";
  oss << " \"git-hash\": " << "\"" << HPC_GIT_HASH_STR << "\",\n";
  oss << " \"g++\": " << "\"g++-" << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__
      << "\",\n";
  oss << " \"hip\": " << "\"" << HIP_VERSION_MAJOR << "." << HIP_VERSION_MINOR << "."
      << HIP_VERSION_PATCH << "\",\n";
  oss << " \"stdc++\": " << "\"" << __cplusplus << "." << "\",\n";
  oss << " \"backend\": " << "\"rocm\",\n";
  oss << " \"built-date\": " << "\"" << __DATE__ << "\",\n";
  oss << " \"built-time\": " << "\"" << __TIME__ << "\",\n";
  oss << " \"_C\": " << "\"" << __FILE__ << "\"\n";
  oss << "}\n";
  // NOLINTEND clang-format on

  return oss.str();
}

TORCH_LIBRARY_FRAGMENT(hpc, m) { m.def("built_json", &built_json); }
