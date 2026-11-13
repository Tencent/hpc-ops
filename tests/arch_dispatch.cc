// Copyright (C) 2026 Tencent.

#include <stdexcept>
#include <string>

#include "src/utils/utils.h"

#define TEST_SUPPORTED_ARCHS (90, 100, 103, 120)

#if !HPC_ARCH_IN(TEST_SUPPORTED_ARCHS)
#error "Unexpected test target"
#endif

static_assert(HPC_ARCH_IN((90)) == (HPC_TARGET_ARCH == 90));
static_assert(HPC_ARCH_IN((90, 100)) == (HPC_TARGET_ARCH == 90 || HPC_TARGET_ARCH == 100));
static_assert(HPC_ARCH_IN((90, 100, 103)) == (HPC_TARGET_ARCH != 120));
static_assert(HPC_ARCH_IN(TEST_SUPPORTED_ARCHS));
static_assert(HPC_ARCH_IN((80, 90, 100, 103, 120)));

namespace hpc {

int runtime_arch = HPC_TARGET_ARCH;

int get_sm_arch() { return runtime_arch; }

[[noreturn]] void throw_arch_not_supported(const char *op, const char *impl_archs) {
  throw std::runtime_error(std::string(op) + ": " + impl_archs);
}

}  // namespace hpc

// Deliberately leave unavailable implementations undefined. The test must link
// even at -O0: unsupported dispatch must not reference their symbols.
int sm90_only();
int sm100_only();
int sm103_only();
int sm120_only();
int subset_only();

#if HPC_ARCH_IN((90))
int sm90_only() { return 90; }
#endif
#if HPC_ARCH_IN((100))
int sm100_only() { return 100; }
#endif
#if HPC_ARCH_IN((103))
int sm103_only() { return 103; }
#endif
#if HPC_ARCH_IN((120))
int sm120_only() { return 120; }
#endif
#if HPC_ARCH_IN((90, 100))
int subset_only() { return HPC_TARGET_ARCH; }
#endif

void require(bool condition) {
  if (!condition) {
    throw std::runtime_error("architecture dispatch check failed");
  }
}

template <int A, int B>
int pair_value() {
  return A + B;
}

int return_from_dispatch() {
  HPC_ARCH_DISPATCH_ANY("return", TEST_SUPPORTED_ARCHS, return HPC_TARGET_ARCH);
}

int main() {
  int value = 0;
  HPC_ARCH_DISPATCH("pairs", 90, value = sm90_only(), 100, value = sm100_only(), 103,
                    value = sm103_only(), 120, value = sm120_only());
  require(value == HPC_TARGET_ARCH);
  require(return_from_dispatch() == HPC_TARGET_ARCH);

  // Template and declaration commas must survive the variadic body unchanged.
  HPC_ARCH_DISPATCH_ANY("shared", TEST_SUPPORTED_ARCHS, {
    int a = 2, b = 3;
    value = pair_value<4, 5>() + a + b;
  });
  require(value == 14);

  value = 0;
  bool rejected = false;
  try {
    HPC_ARCH_DISPATCH_ANY("subset", (90, 100), value = subset_only());
  } catch (const std::runtime_error &error) {
    require(std::string(error.what()) == "subset: 90, 100");
    rejected = true;
  }
  require(rejected == !HPC_ARCH_IN((90, 100)));
  require(value == (HPC_ARCH_IN((90, 100)) ? HPC_TARGET_ARCH : 0));

  // A supported build loaded on the wrong device must reject before side effects.
  hpc::runtime_arch = 0;
  value = 0;
  rejected = false;
  try {
    HPC_ARCH_DISPATCH_ANY("mismatch", TEST_SUPPORTED_ARCHS, ++value);
  } catch (const std::runtime_error &error) {
    require(std::string(error.what()) == "mismatch: 90, 100, 103, 120");
    rejected = true;
  }
  require(rejected && value == 0);

  rejected = false;
  try {
    HPC_ARCH_DISPATCH("pairs", 90, value = sm90_only(), 100, value = sm100_only(), 103,
                      value = sm103_only(), 120, value = sm120_only());
  } catch (const std::runtime_error &error) {
    require(std::string(error.what()) == "pairs: 90, 100, 103, 120");
    rejected = true;
  }
  require(rejected && value == 0);
}
