// UCL SHMEM IBGDA Device API Which is Modified to Improve Performance
// Provides IBGDA device-side enhanced interface under the ucl::shmem namespace
#pragma once

#include "extensions/ibgda_device.cuh"

namespace ucl {
namespace shmem {

// Check whether IBGDA RC is enabled to determine whether to use IBGDA RC extensions
__device__ __forceinline__ bool ibgda_is_enabled() {
    return ibgda_get_state()->num_rc_per_pe > 0;
}

__device__ __forceinline__ uint64_t shmemx_get_p2p_ptr(const uint64_t& ptr, const int& rank, const int& dst_rank) {
    return nvshmemi_get_p2p_ptr(ptr, rank, dst_rank);
}

__device__ __forceinline__ void shmemx_net_amo_nonfetch_add(void *rptr, const int& value, int pe, int qp_id, bool is_local_copy = false) {
    EP_DEVICE_ASSERT(ibgda_is_enabled() && "IBGDA RC is not enabled");
    nvshmemi_ibgda_amo_nonfetch_add(rptr, value, pe, qp_id, is_local_copy);
}

template <bool kAlwaysDoPostSend = false>
__device__ __forceinline__ void shmemx_net_put_nbi_warp(uint64_t req_rptr, uint64_t req_lptr, size_t bytes, 
                                              int dst_pe, int qp_id, int lane_id, int message_idx) {
    EP_DEVICE_ASSERT(ibgda_is_enabled() && "IBGDA RC is not enabled");
    nvshmemi_ibgda_put_nbi_warp<kAlwaysDoPostSend>(req_rptr, req_lptr, bytes, dst_pe, qp_id, lane_id, message_idx);
}

__device__ __forceinline__ void shmemx_net_quiet(int dst_pe, int qp_id) {
    EP_DEVICE_ASSERT(ibgda_is_enabled() && "IBGDA RC is not enabled");
    nvshmemi_ibgda_quiet(dst_pe, qp_id);
}

__device__ __forceinline__ void shmemx_net_rma_p(int *rptr, const int value, int dst_pe, int qp_id, 
                                       uint32_t imm = std::numeric_limits<uint32_t>::max()) {
    EP_DEVICE_ASSERT(ibgda_is_enabled() && "IBGDA RC is not enabled");
    nvshmemi_ibgda_rma_p(rptr, value, dst_pe, qp_id, imm);
}

} // namespace shmem
} // namespace ucl
