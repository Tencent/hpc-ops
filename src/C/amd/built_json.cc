// Copyright (C) 2026 Tencent.
#include <torch/library.h>

#include <string>

static std::string built_json() {
  return "{\"version\":\"" HPC_VERSION_STR "\",\"git-hash\":\"" HPC_GIT_HASH_STR
         "\",\"backend\":\"hip\",\"hip-arch\":\"" HPC_HIP_ARCH_STR "\"}";
}

TORCH_LIBRARY_FRAGMENT(hpc, m) { m.def("built_json", &built_json); }
