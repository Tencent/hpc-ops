// Copyright 2026 hpc-ops authors

#include <cuda.h>

#include "src/fuse_moe/sm100/fuse_moe.h"  // for reduce_async (reused from fp8 path)
#include "src/fuse_moe/sm100/mxfp8/fuse_moe_mxfp8.h"
#include "src/group_gemm/sm100/group_gemm.h"
#include "src/group_gemm/sm100/mxfp8/group_gemm_mxfp8.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace fuse_moe {

// prepack SFA with gather (combines gather + prepack) (gate_up path).
void prepack_mxfp8_sfa_with_gather_async(void *sfa_packed_ptr, const void *x_scale_ptr,
                                         const void *gateup_x_row_map_ptr,
                                         const void *cu_seqlens_ptr, int num_group, int m_total,
                                         int k, int kTileM, cudaStream_t stream, bool use_pdl) {
  constexpr int kSfVec = 32;
  int K_sf = k / kSfVec;
  bool is_smallm = (kTileM <= 128);
  // max_row_tiles: upper bound across all groups; out-of-range CTAs early-exit inside kernel
  int max_row_tiles = (m_total + kTileM - 1) / kTileM;

  if (is_smallm) {
    constexpr int kRowsPerTile = 128;
    constexpr int kSfLanes = 32;
    dim3 grid(num_group, max_row_tiles);
    if (use_pdl) {
      auto kernel =
          group_gemm::kernels::prepack_mxfp8_sfa_with_gather_kernel<kRowsPerTile, kSfLanes, true>;
      cudaLaunchAttribute attr[1];
      attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[0].val.programmaticStreamSerializationAllowed = 1;
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = grid;
      cfg.blockDim = dim3(kSfLanes);
      cfg.dynamicSmemBytes = 0;
      cfg.stream = stream;
      cfg.attrs = attr;
      cfg.numAttrs = 1;
      cudaLaunchKernelEx(&cfg, kernel, reinterpret_cast<const uint8_t *>(x_scale_ptr),
                         reinterpret_cast<uint8_t *>(sfa_packed_ptr),
                         reinterpret_cast<const int *>(gateup_x_row_map_ptr),
                         reinterpret_cast<const int *>(cu_seqlens_ptr), K_sf, kTileM, num_group);
    } else {
      auto kernel =
          group_gemm::kernels::prepack_mxfp8_sfa_with_gather_kernel<kRowsPerTile, kSfLanes, false>;
      kernel<<<grid, kSfLanes, 0, stream>>>(reinterpret_cast<const uint8_t *>(x_scale_ptr),
                                            reinterpret_cast<uint8_t *>(sfa_packed_ptr),
                                            reinterpret_cast<const int *>(gateup_x_row_map_ptr),
                                            reinterpret_cast<const int *>(cu_seqlens_ptr), K_sf,
                                            /*row_stride=*/kTileM, num_group);
    }
  } else {
    constexpr int kRowsPerTile = 256;
    constexpr int kSfLanes = 64;
    dim3 grid(num_group, max_row_tiles);
    if (use_pdl) {
      auto kernel =
          group_gemm::kernels::prepack_mxfp8_sfa_with_gather_kernel<kRowsPerTile, kSfLanes, true>;
      cudaLaunchAttribute attr[1];
      attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[0].val.programmaticStreamSerializationAllowed = 1;
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = grid;
      cfg.blockDim = dim3(kSfLanes);
      cfg.dynamicSmemBytes = 0;
      cfg.stream = stream;
      cfg.attrs = attr;
      cfg.numAttrs = 1;
      cudaLaunchKernelEx(&cfg, kernel, reinterpret_cast<const uint8_t *>(x_scale_ptr),
                         reinterpret_cast<uint8_t *>(sfa_packed_ptr),
                         reinterpret_cast<const int *>(gateup_x_row_map_ptr),
                         reinterpret_cast<const int *>(cu_seqlens_ptr), K_sf, kTileM, num_group);
    } else {
      auto kernel =
          group_gemm::kernels::prepack_mxfp8_sfa_with_gather_kernel<kRowsPerTile, kSfLanes, false>;
      kernel<<<grid, kSfLanes, 0, stream>>>(reinterpret_cast<const uint8_t *>(x_scale_ptr),
                                            reinterpret_cast<uint8_t *>(sfa_packed_ptr),
                                            reinterpret_cast<const int *>(gateup_x_row_map_ptr),
                                            reinterpret_cast<const int *>(cu_seqlens_ptr), K_sf,
                                            /*row_stride=*/kTileM, num_group);
    }
  }
}

void fuse_moe_mxfp8_async(
    void *output_ptr, const void *x_ptr, const void *x_scale_ptr, void *gate_up_input_ptr,
    void *gate_up_input_scale_packed_ptr, void *gate_up_output_ptr, const void *gate_up_weight_ptr,
    const void *gate_up_weight_scale_packed_ptr, void *gate_up_tmas_ptr, void *down_input_ptr,
    void *down_input_scale_packed_ptr, void *down_output_ptr, const void *down_weight_ptr,
    const void *down_weight_scale_packed_ptr, void *down_tmas_ptr, const void *topk_ids_ptr,
    const void *topk_scale_ptr, void *topk_pos_ptr, void *gateup_x_row_map_ptr, void *seqlens_ptr,
    void *cu_seqlens_ptr, void *tiles_ptr, void *cu_tiles_ptr, const void *shared_output_ptr,
    int num_seq, int hidden_size, int intermediate_size, int num_topk, int num_expert_total,
    int num_expert_local, int rank_ep, bool is_fp4, cudaStream_t stream) {
  const int total_num_seq = num_seq * num_topk;
  const int num_seq_per_group_avg = total_num_seq / num_expert_total;
  // Gateup uses cp_async (1SM only), down uses TMA (may use 2SM).
  const int kTileM_gateup = group_gemm::mxfp8_dispatch_kTileM_cp_async(num_seq_per_group_avg);
  const int kTileM_down = group_gemm::mxfp8_dispatch_kTileM(num_seq_per_group_avg, hidden_size);

  // tiles / cu_tiles workspaces are split in half for gateup and down GEMMs.
  void *gateup_tiles_ptr = static_cast<int *>(tiles_ptr);
  void *down_tiles_ptr = static_cast<int *>(tiles_ptr) + num_expert_local;
  void *gateup_cu_tiles_ptr = static_cast<int *>(cu_tiles_ptr);
  void *down_cu_tiles_ptr = static_cast<int *>(cu_tiles_ptr) + (num_expert_local + 1);

  bool use_pdl = true;

  // 1. Route tokens: write topk_pos and gateup_x_row_map; update per-group X/Y TMA
  //    descriptors; compute seqlens / cu_seqlens / tiles / cu_tiles.
  count_and_route_row_mxfp8_async(
      gate_up_input_ptr, gate_up_output_ptr, down_input_ptr, down_output_ptr, topk_ids_ptr,
      topk_pos_ptr, topk_scale_ptr, gateup_x_row_map_ptr, seqlens_ptr, cu_seqlens_ptr,
      gateup_tiles_ptr, gateup_cu_tiles_ptr, down_tiles_ptr, down_cu_tiles_ptr, gate_up_tmas_ptr,
      down_tmas_ptr, num_seq, num_topk, hidden_size, intermediate_size, num_expert_local, rank_ep,
      num_seq_per_group_avg, stream);

  // 2. Prepack SFA for gateup (gather + prepack in one kernel).
  //    Reads x_scale[gateup_x_row_map[irow], :] and writes directly to packed layout.
  prepack_mxfp8_sfa_with_gather_async(gate_up_input_scale_packed_ptr, x_scale_ptr,
                                      gateup_x_row_map_ptr, cu_seqlens_ptr, num_expert_local,
                                      total_num_seq, hidden_size, kTileM_gateup, stream,
                                      /*use_pdl=*/use_pdl);

  // 3. Gateup GEMM via cp_async (N = intermediate_size * 2, K = hidden_size)
  group_gemm::group_gemm_cp_async_mxfp8_async(
      gate_up_output_ptr, x_ptr, gate_up_weight_ptr, gate_up_input_scale_packed_ptr,
      gate_up_weight_scale_packed_ptr, seqlens_ptr, cu_seqlens_ptr, gate_up_tmas_ptr,
      gateup_tiles_ptr, gateup_cu_tiles_ptr, num_expert_local, total_num_seq, intermediate_size * 2,
      hidden_size, num_seq_per_group_avg, /*update_tma=*/false, stream, gateup_x_row_map_ptr,
      num_seq, /*use_pdl=*/use_pdl, /*is_fp4=*/is_fp4);

  // 4. Activation + mxfp8 quant + fused SFA prepack for down.
  //    Writes scale directly to packed layout, skipping the separate prepack kernel.
  const int *valid_row_range_ptr = static_cast<const int *>(cu_seqlens_ptr) + num_expert_local;
  constexpr int kSfVec = 32;
  int k_sf_tiles_down = (intermediate_size / kSfVec + 3) / 4;
  act_mul_and_mxfp8_quant_async(down_input_ptr, down_input_scale_packed_ptr, gate_up_output_ptr,
                                valid_row_range_ptr, cu_seqlens_ptr, down_cu_tiles_ptr,
                                num_expert_local, total_num_seq, intermediate_size, kTileM_down,
                                k_sf_tiles_down, stream, use_pdl);

  // 5. Down GEMM (N = hidden_size, K = intermediate_size).
  group_gemm::group_gemm_mxfp8_async(
      down_output_ptr, down_input_ptr, down_weight_ptr, down_input_scale_packed_ptr,
      down_weight_scale_packed_ptr, seqlens_ptr, cu_seqlens_ptr, down_tmas_ptr, down_tiles_ptr,
      down_cu_tiles_ptr, num_expert_local, total_num_seq, hidden_size, intermediate_size,
      num_seq_per_group_avg, /*update_tma=*/false, stream, /*use_pdl=*/use_pdl,
      /*is_fp4=*/is_fp4);

  // 6. Reduce: scatter-add by topk_pos with topk_scale weights, optional shared_output add.
  reduce_async(output_ptr, down_output_ptr, topk_pos_ptr, topk_scale_ptr, shared_output_ptr,
               total_num_seq, num_seq, hidden_size, num_topk, use_pdl, stream);
}

}  // namespace fuse_moe
}  // namespace hpc
