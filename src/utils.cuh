
#define kWarpSize 32
#define UP_DIV(X, Y) (((X) + (Y) - 1) / (Y))

namespace hpc {
namespace utils {

__device__ __forceinline__ void load_16B_from_bf16_to_float(
    float* dst_ptr, __nv_bfloat16* src_ptr) {
  __nv_bfloat162 data[4];
  *((int4*)(&data[0])) = *((int4*)(src_ptr));

#pragma unroll
  for (int i = 0; i < 4; i++) {
    *((float2*)(&dst_ptr[2 * i])) = __bfloat1622float2(data[i]);
  }
}

__device__ __forceinline__ void store_16B_to_bf16_from_float(
    __nv_bfloat16* dst_ptr, float* src_ptr) {
  __nv_bfloat162 data[4];

#pragma unroll
  for (int i = 0; i < 4; i++) {
    data[i] = __float22bfloat162_rn(*((float2*)(&src_ptr[2 * i])));
  }
  *((int4*)(dst_ptr)) = *((int4*)(&data[0]));
}

__device__ __forceinline__ void store_8B_to_fp8e4m3_from_float(
    __nv_fp8_e4m3* dst_ptr, float* src_ptr) {
  *((__nv_fp8x4_e4m3*)(dst_ptr)) = __nv_fp8x4_e4m3(*((float4*)(src_ptr)));
  *((__nv_fp8x4_e4m3*)(&dst_ptr[4])) =
      __nv_fp8x4_e4m3(*((float4*)(&src_ptr[4])));
}

__device__ __forceinline__ void store_16B_to_float_from_float(float* dst_ptr,
                                                              float* src_ptr) {
  *((int4*)(dst_ptr)) = *((int4*)(src_ptr));
}

}  // namespace utils
}  // namespace hpc
