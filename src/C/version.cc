// Copyright (C) 2026 Tencent.

#include <tvm/ffi/function.h>

#include <string>

#ifndef HPC_VERSION_STR
#define HPC_VERSION_STR "unknown"
#endif

static std::string version() { return HPC_VERSION_STR; }

TVM_FFI_DLL_EXPORT_TYPED_FUNC(version, version);
