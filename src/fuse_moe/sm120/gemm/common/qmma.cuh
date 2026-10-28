// Copyright (C) 2026 Tencent.

#pragma once

#include <cute/tensor.hpp>

namespace hpc::fuse_moe::sm120::gemm {
namespace common {

// mma.sync m16n8k32 E4M3 x {E2M1, E4M3} with UE8M0 scales, taking each
// operand's scales as the packed word of four k-blocks and selecting byte kKb in
// the instruction. The CuTe atom takes a single scale byte, which costs a shift
// and a mask per MMA and serializes the MMAs on the few registers holding the
// extracted bytes.
template <int kKb, class ElementB>
CUTE_DEVICE void qmma_bytesel(float (&d)[4], uint32_t const (&a)[4], uint32_t const (&b)[2],
                              uint32_t sfa, uint32_t sfb) {
  if constexpr (cute::is_same_v<ElementB, cute::float_e2m1_t>) {
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1."
        "f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%12, %13}, "
        "{%11}, {%12, %13};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sfa), "r"(sfb),
          "n"(kKb), "n"(0));
  } else {
    static_assert(cute::is_same_v<ElementB, cute::float_e4m3_t>);
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3."
        "f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%12, %13}, "
        "{%11}, {%12, %13};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sfa), "r"(sfb),
          "n"(kKb), "n"(0));
  }
}

// E4M3 x ElementB mainloop for fragments from the SM120 block-scaled atom.
// sfa_words holds one packed scale word per row block and sfb_words one per
// column block, four k-blocks to a word.
template <int kLive, class ElementB, class TA, class TB, class TC, class SFA, class SFB>
CUTE_DEVICE void qmma_live_rows_bytesel(TA const& A, SFA const& sfa_words, TB const& B,
                                        SFB const& sfb_words, TC& C) {
  constexpr int kBlocksK = decltype(cute::size<2>(A))::value;
  static_assert(kBlocksK <= 4, "one packed scale word covers four k-blocks");
  cute::for_each(cute::make_int_sequence<kBlocksK>{}, [&](auto kb) {
    constexpr int k = decltype(kb)::value;
    CUTE_UNROLL
    for (int m = 0; m < kLive; ++m) {
      auto a = cute::recast<uint32_t const>(A(cute::_, m, k));
      uint32_t a_regs[4] = {a(0), a(1), a(2), a(3)};
      CUTE_UNROLL
      for (int n = 0; n < cute::size<1>(B); ++n) {
        auto b = cute::recast<uint32_t const>(B(cute::_, n, k));
        uint32_t b_regs[2] = {b(0), b(1)};
        auto c = C(cute::_, m, n);
        float d[4] = {c(0), c(1), c(2), c(3)};
        qmma_bytesel<k, ElementB>(d, a_regs, b_regs, static_cast<uint32_t>(sfa_words[m]),
                                  static_cast<uint32_t>(sfb_words[n]));
        c(0) = d[0];
        c(1) = d[1];
        c(2) = d[2];
        c(3) = d[3];
      }
    }
  });
}

}  // namespace common
}  // namespace hpc::fuse_moe::sm120::gemm
