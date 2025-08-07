#include <torch/library.h>

#include <sstream>

#ifndef HPC_VERSION_STR
#define HPC_VERSION_STR "unknown"
#endif

static const std::string version() { return HPC_VERSION_STR; }

static const std::string built_json() {
  std::ostringstream oss;

  // clang-format off
  oss << "{" << "\n";
  oss << " \"built-date\": " << "\"" << __DATE__ << "\",\n";
  oss << " \"built-time\": " << "\"" << __TIME__ << "\",\n";
  oss << " \"_C\": " << "\"" << __FILE__ << "\",\n";
  oss << " \"version\": " << "\"" << HPC_VERSION_STR << "\",\n";
  oss << " \"compiler\": " << "\"g++-" << __GNUC__ << "." << __GNUC_MINOR__ << "."
      << __GNUC_PATCHLEVEL__ << "\",\n";
  oss << " \"glibc\": " << "\"" << __GLIBC__ << "." << __GLIBC_MINOR__ << "\",\n";
  oss << "}\n";
  // clang-format on

  return oss.str();
}

TORCH_LIBRARY(hpc, m) { m.def("version", &version).def("built_json", built_json); }
