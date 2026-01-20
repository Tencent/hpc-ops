// Copyright (C) 2026 Tencent.

#include <cutlass/version.h>
#include <torch/library.h>

#include <cub/version.cuh>
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
  oss << " \"nvcc\": " << "\"nvcc-" << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << "."
      << __CUDACC_VER_BUILD__ << "\",\n";
  oss << " \"stdc++\": " << "\"" << __cplusplus << "." << "\",\n";
  oss << " \"glibc\": " << "\"" << __GLIBC__ << "." << __GLIBC_MINOR__ << "\",\n";
  oss << " \"cub\": " << "\"" << CUB_MAJOR_VERSION << "." << CUB_MINOR_VERSION << "."
      << CUB_SUBMINOR_VERSION << "\",\n";
  oss << " \"cutlass\": " << "\"" << CUTLASS_MAJOR << "." << CUTLASS_MINOR << "." << CUTLASS_PATCH
      << "\",\n";
  oss << " \"built-date\": " << "\"" << __DATE__ << "\",\n";
  oss << " \"built-time\": " << "\"" << __TIME__ << "\",\n";
  oss << " \"built-ts\": " << "\"" << __TIMESTAMP__ << "." << "\",\n";
  oss << " \"_C\": " << "\"" << __FILE__ << "\"\n";
  oss << "}\n";
  // NOLINTEND clang-format on

  return oss.str();
}

TORCH_LIBRARY_FRAGMENT(hpc, m) { m.def("built_json", &built_json); }
