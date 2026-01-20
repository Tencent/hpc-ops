// Copyright (C) 2026 Tencent.

#include <torch/library.h>

#include <sstream>
#include <string>

#ifndef HPC_VERSION_STR
#define HPC_VERSION_STR "unknown"
#endif

static const std::string version() { return HPC_VERSION_STR; }

TORCH_LIBRARY_FRAGMENT(hpc, m) { m.def("version", &version); }
