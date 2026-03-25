/*
 * UCL SHMEM API - Device-Host Proxy Channel
 * This file enhances NVSHMEM proxy channel types into UCL shmem namespace
 */

#ifndef UCL_SHMEM_DEVICE_HOST_PROXY_CHANNEL_H
#define UCL_SHMEM_DEVICE_HOST_PROXY_CHANNEL_H

#include "device_host/nvshmem_proxy_channel.h"

namespace ucl {
namespace shmem {

// Re-export proxy channel types
using channel_bounce_buffer_t = ::channel_bounce_buffer_t;
using base_request_t = ::base_request_t;
using put_dma_request_0_t = ::put_dma_request_0_t;
using put_dma_request_1_t = ::put_dma_request_1_t;
using put_dma_request_2_t = ::put_dma_request_2_t;
using put_inline_request_0_t = ::put_inline_request_0_t;
using put_inline_request_1_t = ::put_inline_request_1_t;
using amo_request_0_t = ::amo_request_0_t;
using amo_request_1_t = ::amo_request_1_t;
using amo_request_2_t = ::amo_request_2_t;
using amo_request_3_t = ::amo_request_3_t;

} // namespace shmem
} // namespace ucl

#endif // UCL_SHMEM_DEVICE_HOST_PROXY_CHANNEL_H
