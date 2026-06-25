"""Standalone CuteDSL backend for the FP8 paged-prefill kernel.

This file intentionally does not keep a scalar or math-only fallback.  The user
visible contract must match the current generated-kernel path, not just the Python
reference equation:

* Q/K/V are E4M3 FP8 payloads.
* K/V cache tensors are paged views over the same backing allocation as the
  packed K-scale tail rows.
* K/V tiles are loaded through the page table; no wrapper-side gather,
  contiguous copy, or unpacking is allowed.
* Q scale and K scale are applied to the QK accumulator before row-max/softmax.
* Softmax P is materialized as E4M3 payload before PV; the denominator still
  uses the unrounded exp2 value, matching TmemP.h.
* V scale is applied in the final correction/epilogue.

The old row-CTA implementation was removed because it did not mirror the
generated-kernel dataflow: it decoded FP8 scalars in CUDA cores, did not use TCGEN,
did not stage K scale like SmemSageAttnSfsK, and did not round P to E4M3 before
PV.  Leaving that path callable would hide implementation drift.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Type

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Float32, Int32, Int64, Uint32

# fmha_helpers is vendored under attention/utils/ so the hpc-ops wheel is
# self-contained and does not require CUTLASS_HOME / CUTEDSL_EXAMPLES_DIR
# to be set at runtime. See utils/LICENSE-NVIDIA-CUTLASS.txt for licensing.
from .utils import fmha_helpers as fmha_utils


def make_thread_cooperative_group(size: int):
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, size)


class BlackwellFusedMultiHeadAttentionForward:
    """Minimal FMHA base copied into delivery to avoid importing the development package."""

    def __init__(
        self,
        qk_acc_dtype: Type[cutlass.Numeric],
        pv_acc_dtype: Type[cutlass.Numeric],
        mma_tiler: Tuple[int, int, int],
        is_persistent: bool,
        mask_type: fmha_utils.MaskEnum,
    ):
        """Initializes the configuration for a Blackwell Fused Multi-Head Attention (FMHA) kernel.

        This configuration includes several key aspects:

        1.  Data Type Settings:
            - qk_acc_dtype: Data type for Q*K^T matrix multiplication accumulator
            - pv_acc_dtype: Data type for P*V matrix multiplication accumulator

        2.  MMA Instruction Settings:
            - mma_tiler: The (M, N, K) shape of the MMA instruction unit
            - qk_mma_tiler: MMA shape for Q*K^T computation
            - pv_mma_tiler: MMA shape for P*V computation

        3.  Kernel Execution Mode:
            - is_persistent: Boolean indicating whether to use persistent kernel mode
            - mask_type: Specifies the type of mask to use (no mask, residual mask, or causal mask)
            - window_size_left/right: Sliding window size for attention masking

        :param qk_acc_dtype: Data type for Q*K^T matrix multiplication accumulator
        :type qk_acc_dtype: Type[cutlass.Numeric]
        :param pv_acc_dtype: Data type for P*V matrix multiplication accumulator
        :type pv_acc_dtype: Type[cutlass.Numeric]
        :param mma_tiler: The (M, N, K) shape of the MMA instruction
        :type mma_tiler: Tuple[int, int, int]
        :param is_persistent: Whether to use persistent kernel mode
        :type is_persistent: bool
        :param mask_type: Type of mask to use
        :type mask_type: fmha_utils.MaskEnum
        :param window_size_left: Left-side sliding window size for attention masking
        :type window_size_left: int
        :param window_size_right: Right-side sliding window size for attention masking
        :type window_size_right: int
        """

        self.qk_acc_dtype = qk_acc_dtype
        self.pv_acc_dtype = pv_acc_dtype
        self.cta_tiler = (
            2 * mma_tiler[0],  # 2 Q tile per CTA
            mma_tiler[1],
            mma_tiler[2],
        )
        self.qk_mma_tiler = mma_tiler
        self.pv_mma_tiler = (
            mma_tiler[0],
            mma_tiler[2],
            mma_tiler[1],
        )
        self.cluster_shape_mn = (1, 1)
        self.is_persistent = is_persistent
        self.mask_type = mask_type

        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.epilogue_warp_id = 14
        self.empty_warp_id = 15
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.epilogue_warp_id,
                self.empty_warp_id,
            )
        )

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_warp,
        )

        self.tmem_s0_offset = 0
        self.tmem_s1_offset = 128
        self.tmem_o0_offset = 256
        self.tmem_o1_offset = 384
        self.tmem_p0_offset = 32
        self.tmem_p1_offset = 160

        # vec buffer for row_max & row_sum
        self.tmem_vec0_offset = 0
        self.tmem_vec1_offset = 128

        self.num_regs_softmax = 192
        self.num_regs_correction = 96
        self.num_regs_other = 32

        self.buffer_align_bytes = 1024

        num_warps_per_warpgroup = 4
        self.softmax_warpgroup_count = (
            len((*self.softmax0_warp_ids, *self.softmax1_warp_ids)) // num_warps_per_warpgroup
        )

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        self.q_stage = 2
        self.kv_stage = 4 if self.q_dtype.width == 8 else 3
        self.acc_stage = 1
        self.softmax_corr_stage = 1
        self.mma_corr_stage = 2
        self.mma_softmax_stage = 1
        self.epi_stage = 2


LOG2E = math.log2(math.e)
LOG2_E4M3_MAX = math.log2(448.0)
# Default p_scale used when caller does not pass an explicit value. Set to
# 256 (rather than the e4m3 saturation cap 448) to keep ~0.57x headroom in
# the fp8 P bucket, lowering saturation/clamp risk on long-tail rows.
DEFAULT_P_SCALE = 256.0
DEFAULT_P_SCALE_LOG2 = math.log2(DEFAULT_P_SCALE)
INV_SQRT_HEAD_DIM_128 = 1.0 / math.sqrt(128.0)
SCALE_SOFTMAX_LOG2_HEAD_DIM_128 = INV_SQRT_HEAD_DIM_128 * LOG2E


@dsl_user_op
def _cvt_float4_to_e4m3_u32(
    a: Float32,
    b: Float32,
    c: Float32,
    d: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Pack four FP32 values into one E4M3x4 word using the native PTX path."""

    return Uint32(
        llvm.inline_asm(
            Uint32.mlir_type,
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                Float32(c).ir_value(loc=loc, ip=ip),
                Float32(d).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t"
            ".reg .b16 lo;\n\t"
            ".reg .b16 hi;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n\t"
            "mov.b32 $0, {lo, hi};\n\t"
            "}",
            "=r,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _storage_data_ptr(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def _as_cute_tensor(
    tensor: torch.Tensor,
    element_type: Type[cutlass.Numeric],
    *,
    assumed_align: int = 16,
    leading_dim: int = -1,
    fully_dynamic: bool = False,
) -> cute.Tensor:
    dlpack_tensor = tensor
    if tensor.dtype == torch.float8_e4m3fn:
        # PyTorch DLPack still does not reliably advertise float8; this is a
        # zero-copy storage view.  The CuteDSL tensor element type is set below.
        dlpack_tensor = tensor.view(torch.uint8)
    cute_tensor = from_dlpack(
        dlpack_tensor,
        assumed_align=assumed_align,
        enable_tvm_ffi=True,
    )
    cute_tensor.element_type = element_type
    if leading_dim == -1:
        leading_dim = tensor.dim() - 1
    return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)


def _output_element_type(output: torch.Tensor) -> Type[cutlass.Numeric]:
    if output.dtype == torch.bfloat16:
        return cutlass.BFloat16
    if output.dtype == torch.float16:
        return cutlass.Float16
    if output.dtype == torch.float32:
        return cutlass.Float32
    raise TypeError(f"unsupported output dtype: {output.dtype}")


@dataclass(frozen=True)
class Fp8PagedPrefillConfig:
    """Static shape choices for the FP8 paged-prefill kernel."""

    threads_per_cta: int = 512
    head_dim: int = 128
    block_size: int = 64
    tile_q: int = 128
    tile_kv: int = 128
    is_persistent: bool = True


class Fp8PagedPrefillCuteDSL(BlackwellFusedMultiHeadAttentionForward):
    """CuteDSL FP8 paged-prefill runner."""

    def __init__(self, config: Fp8PagedPrefillConfig = Fp8PagedPrefillConfig()):
        super().__init__(
            qk_acc_dtype=cutlass.Float32,
            pv_acc_dtype=cutlass.Float32,
            mma_tiler=(config.tile_q, config.tile_kv, config.head_dim),
            is_persistent=config.is_persistent,
            mask_type=fmha_utils.MaskEnum.WINDOW_MASK_INFERENCE,
        )
        self.fp8_config = config
        self._compiled_fmha = {}

    def _setup_attributes(self):
        super()._setup_attributes()
        # Match the generated persistent kernel archive:
        #   SmemQ[2][16384], SmemKv[3][16384], SmemSageAttnSfsK[6][128].
        self.q_stage = 2
        self.kv_stage = 2
        self.sage_k_stage = 7
        self.num_regs_softmax = 200
        self.num_regs_correction = 80

    def prepare_runtime_tensors(
        self,
        q: torch.Tensor,
        kcache: torch.Tensor,
        vcache: torch.Tensor,
        qscale: torch.Tensor,
        kscale: torch.Tensor,
        vscale: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        block_ids: torch.Tensor,
        seqlens_kvcache: torch.Tensor,
        output: torch.Tensor,
        p_scale_log2: torch.Tensor,
    ):
        """Create zero-copy CuteDSL views for the packed-cache ABI.

        `kscale` stays the FP8 tail-row view owned by the cache allocation.  The
        float32 tensor below is a storage reinterpretation only; it is not a
        materialized unpack, gather, or contiguous copy.

        ``p_scale_log2`` is a fp32 ``[num_head_q]`` tensor filled with the
        fixed internal ``log2(DEFAULT_P_SCALE)`` (= ``log2(256)``), so the
        kernel always reads from the same indirection.
        """

        kscale_words = kscale.view(torch.float32)
        # Bulk K/V/Q/output payloads go through TMA / 16B vector loads and
        # keep ``assumed_align=16``. Metadata tensors are scalar-loaded; use
        # natural (4B) alignment so callers can pass non-16B-aligned buffers.
        return (
            _as_cute_tensor(q, cutlass.Float8E4M3FN),
            _as_cute_tensor(kcache, cutlass.Float8E4M3FN),
            _as_cute_tensor(vcache, cutlass.Float8E4M3FN),
            _as_cute_tensor(qscale, cutlass.Float32, assumed_align=4, fully_dynamic=True),
            _as_cute_tensor(kscale_words, cutlass.Float32, assumed_align=4, fully_dynamic=True),
            _as_cute_tensor(vscale, cutlass.Float32, assumed_align=4, fully_dynamic=True),
            _as_cute_tensor(cu_seqlens_q, Int32, assumed_align=4, fully_dynamic=True),
            _as_cute_tensor(block_ids, Int32, assumed_align=4, fully_dynamic=True),
            _as_cute_tensor(seqlens_kvcache, Int32, assumed_align=4, fully_dynamic=True),
            _as_cute_tensor(output, _output_element_type(output)),
            _as_cute_tensor(p_scale_log2, cutlass.Float32, assumed_align=4, fully_dynamic=True),
        )

    def __call__(
        self,
        q: torch.Tensor,
        kcache: torch.Tensor,
        vcache: torch.Tensor,
        qscale: torch.Tensor,
        kscale: torch.Tensor,
        vscale: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        block_ids: torch.Tensor,
        seqlens_kvcache: torch.Tensor,
        max_seqlens_q: int,
        output: torch.Tensor,
        p_scale_log2: torch.Tensor,
    ) -> torch.Tensor:
        tensors = self.prepare_runtime_tensors(
            q,
            kcache,
            vcache,
            qscale,
            kscale,
            vscale,
            cu_seqlens_q,
            block_ids,
            seqlens_kvcache,
            output,
            p_scale_log2,
        )
        # Launch on the caller's current CUDA stream (matches the C++
        # ``at::cuda::getCurrentCUDAStream`` convention used elsewhere).
        _torch_stream = torch.cuda.current_stream(q.device)
        stream = cuda.CUstream(_torch_stream.cuda_stream)
        compile_key = (
            self.fp8_config,
            q.dtype,
            kcache.dtype,
            vcache.dtype,
            kscale.dtype,
            output.dtype,
            int(q.shape[1]),
            int(kcache.shape[1]),
            int(kcache.shape[2]),
            int(kscale.shape[1]),
            tuple(s == 0 for s in q.stride()),
            tuple(s == 0 for s in kcache.stride()),
            tuple(s == 0 for s in vcache.stride()),
            tuple(s == 0 for s in kscale.stride()),
            tuple(s == 0 for s in output.stride()),
            int(kscale.stride(0) // 4),
        )
        compiled = self._compiled_fmha.get(compile_key)
        if compiled is None:
            compiled = cute.compile(
                self.launch_fmha_paged,
                *tensors,
                Int32(cu_seqlens_q.numel() - 1),
                Int32(max_seqlens_q),
                Int32(q.shape[1]),
                Int32(kcache.shape[2]),
                Int32(kscale.stride(0) // 4),
                Int32(kscale.stride(1) // 4),
                Int32(kscale.stride(2) // 4),
                stream,
                options="--enable-tvm-ffi",
            )
            self._compiled_fmha[compile_key] = compiled
        compiled(
            *tensors,
            Int32(cu_seqlens_q.numel() - 1),
            Int32(max_seqlens_q),
            Int32(q.shape[1]),
            Int32(kcache.shape[2]),
            Int32(kscale.stride(0) // 4),
            Int32(kscale.stride(1) // 4),
            Int32(kscale.stride(2) // 4),
            stream,
        )
        return output

    @cute.jit
    def launch_fmha_paged(
        self,
        q: cute.Tensor,
        kcache: cute.Tensor,
        vcache: cute.Tensor,
        qscale: cute.Tensor,
        kscale_words: cute.Tensor,
        vscale: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        block_ids: cute.Tensor,
        seqlens_kvcache: cute.Tensor,
        output: cute.Tensor,
        p_scale_log2: cute.Tensor,
        batch: Int32,
        max_seqlens_q: Int32,
        num_head_q: Int32,
        num_head_kv: Int32,
        kscale_page_stride_words: Int32,
        kscale_row_stride_words: Int32,
        kscale_head_kv_stride_words: Int32,
        stream: cuda.CUstream,
    ):
        head_dim = Int32(self.fp8_config.head_dim)
        h_r = num_head_q // num_head_kv
        mQ_qdl, _ = self.make_varlen_q_o_tensors(
            q,
            output,
            batch,
            max_seqlens_q,
            num_head_q,
            num_head_kv,
        )
        mK_pdl, mV_pdl = self.make_paged_kv_physical_tensors(
            kcache,
            vcache,
            h_r,
            num_head_kv,
        )

        self.q_dtype = q.element_type
        self.k_dtype = kcache.element_type
        self.v_dtype = vcache.element_type
        self.o_dtype = output.element_type

        self.tile_sched_params, grid = fmha_utils.compute_grid(
            cute.shape((max_seqlens_q, head_dim, ((h_r, num_head_kv), batch))),
            self.cta_tiler,
            self.is_persistent,
        )
        if cutlass.const_expr(self.is_persistent):
            # Match the generated kernel's flattened logical CTA grid instead of limiting
            # the launch to one CTA per SM.  The scheduler still maps blockIdx.x
            # through the same linear work layout, so each CTA owns one Q tile.
            grid = (cute.size(self.tile_sched_params.problem_shape_mbh), 1, 1)

        self.q_major_mode = utils.LayoutEnum.ROW_MAJOR.mma_major_mode()
        self.k_major_mode = tcgen05.OperandMajorMode.K
        self.v_major_mode = tcgen05.OperandMajorMode.MN
        self.o_layout = utils.LayoutEnum.ROW_MAJOR

        if cutlass.const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of q is not supported")
        if cutlass.const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if cutlass.const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.ONE
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.qk_mma_tiler[:2],
        )
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.pv_mma_tiler[:2],
            p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )
        self.epi_tile = self.pv_mma_tiler[:2]

        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.q_dtype,
            self.q_stage,
        )
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.k_dtype,
            self.kv_stage,
        )
        p_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.q_dtype,
            self.acc_stage,
        )
        v_smem_layout_staged = sm100_utils.make_smem_layout_b(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.v_dtype,
            self.kv_stage,
        )

        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)

        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ_qdl,
            q_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # The logical 128-token KV tile is materialized as two 64-token page TMA
        # copies into the full 128xD smem stage, matching SmemKv.h.
        k_page_mma_tiler = (
            self.qk_mma_tiler[0],
            self.fp8_config.block_size,
            self.fp8_config.head_dim,
        )
        qk_page_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            k_page_mma_tiler[:2],
        )
        k_page_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_page_tiled_mma,
            k_page_mma_tiler,
            self.k_dtype,
            1,
        )
        k_page_smem_layout = cute.select(k_page_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_k_page, tma_tensor_k_page = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mK_pdl,
            k_page_smem_layout,
            k_page_mma_tiler,
            qk_page_tiled_mma,
            (1, 1, 1),
        )

        v_page_mma_tiler = (
            self.pv_mma_tiler[0],
            self.pv_mma_tiler[1],
            self.fp8_config.block_size,
        )
        pv_page_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            v_page_mma_tiler[:2],
            p_source,
        )
        v_page_smem_layout_staged = sm100_utils.make_smem_layout_b(
            pv_page_tiled_mma,
            v_page_mma_tiler,
            self.v_dtype,
            1,
        )
        v_page_smem_layout = cute.select(v_page_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_v_page, tma_tensor_v_page = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mV_pdl,
            v_page_smem_layout,
            v_page_mma_tiler,
            pv_page_tiled_mma,
            (1, 1, 1),
        )

        self.tma_copy_q_bytes = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        self.tma_copy_kv_bytes = cute.size_in_bytes(
            self.k_dtype,
            cute.select(k_smem_layout_staged, mode=[0, 1, 2]),
        )

        @cute.struct
        class SharedStorage:
            load_q_mbar_ptr: cute.struct.MemRange[Int64, self.q_stage * 2]
            load_kv_mbar_ptr: cute.struct.MemRange[Int64, self.kv_stage * 2]
            load_sage_k_mbar_ptr: cute.struct.MemRange[Int64, self.sage_k_stage * 2]
            mma_s0_mbar_ptr: cute.struct.MemRange[Int64, self.mma_softmax_stage * 2]
            mma_s1_mbar_ptr: cute.struct.MemRange[Int64, self.mma_softmax_stage * 2]
            s0_corr_mbar_ptr: cute.struct.MemRange[Int64, self.softmax_corr_stage * 2]
            s1_corr_mbar_ptr: cute.struct.MemRange[Int64, self.softmax_corr_stage * 2]
            s0_s1_sequence_mbar_ptr: cute.struct.MemRange[Int64, self.softmax_warpgroup_count]
            mma_corr_mbar_ptr: cute.struct.MemRange[Int64, self.mma_corr_stage * 2]
            tmem_dealloc_mbar_ptr: cute.struct.MemRange[Int64, 1]
            tmem_holding_buf: Int32
            sSfsQ: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.fp8_config.tile_q * self.q_stage],
                128,
            ]
            sSfsK: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.fp8_config.tile_kv * self.sage_k_stage],
                128,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(k_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel_sage_paged(
            qk_tiled_mma,
            pv_tiled_mma,
            qk_page_tiled_mma,
            pv_page_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k_page,
            tma_tensor_k_page,
            tma_atom_v_page,
            tma_tensor_v_page,
            output,
            qscale,
            kscale_words,
            vscale,
            cu_seqlens_q,
            block_ids,
            seqlens_kvcache,
            q_smem_layout_staged,
            k_smem_layout_staged,
            k_page_smem_layout_staged,
            p_tmem_layout_staged,
            v_smem_layout_staged,
            v_page_smem_layout_staged,
            self.tile_sched_params,
            h_r,
            num_head_kv,
            kscale_page_stride_words,
            kscale_row_stride_words,
            kscale_head_kv_stride_words,
            p_scale_log2,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def load_kv_page_indices(
        self,
        block_ids: cute.Tensor,
        batch_idx: Int32,
        kv_tile_start: Int32,
        seqlen_kv: Int32,
    ):
        """Load the two physical page ids backing one 128-token KV tile."""

        page_idx_ub = ((seqlen_kv + Int32(63)) >> Int32(6)) - Int32(1)
        page_idx_in_seq0 = kv_tile_start >> Int32(6)
        page_idx_in_seq0 = cutlass.min(page_idx_in_seq0, page_idx_ub)
        page_idx_in_seq1 = cutlass.min(page_idx_in_seq0 + Int32(1), page_idx_ub)
        return block_ids[batch_idx, page_idx_in_seq0], block_ids[batch_idx, page_idx_in_seq1]

    @cute.jit
    def copy_paged_k_tile_tma(
        self,
        qk_page_tiled_mma: cute.TiledMma,
        tma_atom_k_page: cute.CopyAtom,
        mK_pdl: cute.Tensor,
        sK_full: cute.Tensor,
        head_kv: Int32,
        page_idx0: Int32,
        page_idx1: Int32,
        k_page_smem_layout_staged: cute.ComposedLayout,
        stage_idx: Int32,
        tma_bar_ptr: cute.Pointer,
    ):
        """Copy one 128-token logical K tile as two 64-token TMA page copies."""

        qk_page_thr_mma = qk_page_tiled_mma.get_slice(0)
        gK_page = cute.flat_divide(
            mK_pdl,
            cute.select((128, 64, self.fp8_config.head_dim), mode=[1, 2]),
        )
        tSgK = qk_page_thr_mma.partition_B(gK_page)
        for slot in cutlass.range_constexpr(2):
            if cutlass.const_expr(slot == 0):
                page_idx = page_idx0
            else:
                page_idx = page_idx1
            page_offset = stage_idx * Int32(16384) + Int32(slot * 8192)
            sK_page_ptr = cute.recast_ptr(
                sK_full.iterator + page_offset,
                k_page_smem_layout_staged.inner,
            )
            sK_page = cute.make_tensor(
                sK_page_ptr,
                k_page_smem_layout_staged.outer,
            )
            tKsK, tKgK_pdl = cute.nvgpu.cpasync.tma_partition(
                tma_atom_k_page,
                0,
                cute.make_layout(1),
                cute.group_modes(sK_page, 0, 3),
                cute.group_modes(tSgK, 0, 3),
            )
            tKgK = tKgK_pdl[None, None, 0, ((Int32(0), head_kv), page_idx)]
            cute.copy(
                tma_atom_k_page,
                tKgK[None, 0],
                tKsK[None, 0],
                tma_bar_ptr=tma_bar_ptr,
            )

    @cute.jit
    def copy_paged_v_tile_tma(
        self,
        pv_page_tiled_mma: cute.TiledMma,
        tma_atom_v_page: cute.CopyAtom,
        mV_pdl: cute.Tensor,
        sV_full: cute.Tensor,
        head_kv: Int32,
        page_idx0: Int32,
        page_idx1: Int32,
        v_page_smem_layout_staged: cute.ComposedLayout,
        stage_idx: Int32,
        tma_bar_ptr: cute.Pointer,
    ):
        """Copy one 128-token logical V tile as two 64-token TMA page copies."""

        pv_page_thr_mma = pv_page_tiled_mma.get_slice(0)
        gV_page = cute.flat_divide(
            mV_pdl,
            cute.select((128, self.fp8_config.head_dim, 64), mode=[1, 2]),
        )
        tSgV = pv_page_thr_mma.partition_B(gV_page)
        for slot in cutlass.range_constexpr(2):
            if cutlass.const_expr(slot == 0):
                page_idx = page_idx0
            else:
                page_idx = page_idx1
            page_offset = stage_idx * Int32(16384) + Int32(slot * 8192)
            sV_page_ptr = cute.recast_ptr(
                sV_full.iterator + page_offset,
                v_page_smem_layout_staged.inner,
            )
            sV_page = cute.make_tensor(
                sV_page_ptr,
                v_page_smem_layout_staged.outer,
            )
            tVsV, tVgV_pdl = cute.nvgpu.cpasync.tma_partition(
                tma_atom_v_page,
                0,
                cute.make_layout(1),
                cute.group_modes(sV_page, 0, 3),
                cute.group_modes(tSgV, 0, 3),
            )
            tVgV = tVgV_pdl[None, None, 0, ((Int32(0), head_kv), page_idx)]
            cute.copy(
                tma_atom_v_page,
                tVgV[None, 0],
                tVsV[None, 0],
                tma_bar_ptr=tma_bar_ptr,
            )

    @cute.jit
    def load_sage_k_scale_tile(
        self,
        kscale_words: cute.Tensor,
        smem_sfs_k: cute.Tensor,
        head_kv: Int32,
        page_idx0: Int32,
        page_idx1: Int32,
        kv_tile_start: Int32,
        seqlen_kv: Int32,
        num_head_kv: Int32,
        kscale_page_stride_words: Int32,
        kscale_row_stride_words: Int32,
        kscale_head_kv_stride_words: Int32,
    ):
        """Load K scales using fully dynamic strides so the caller can pass
        either the original NHD-packed layout or an HND-permuted view of the
        same underlying allocation.

        For head_dim=128 and mNumEltsPerSageAttnBlkK=1:

        * numTokensPerScaleRowK = head_dim / 4 = 32  (innermost = lane_col)
        * scaleRow stride         = kscale_row_stride_words
        * headKv   stride         = kscale_head_kv_stride_words
        * page     stride         = kscale_page_stride_words

        NHD baseline: row stride = num_head_kv * 32, head_kv stride = 32.
        HND view:     row stride = 32,               head_kv stride = (block+scale_rows)*32.
        The formula below works for both because each factor is read from
        kscale.stride() at host side.
        """

        lane_col = cute.arch.lane_idx()
        atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.ALWAYS),
            cutlass.Float32,
            num_bits_per_copy=cutlass.Float32.width,
        )

        for loop_offset in cutlass.range_constexpr(4):
            col = lane_col + Int32(loop_offset * 32)
            token_idx = kv_tile_start + col
            if token_idx < seqlen_kv:
                if cutlass.const_expr(loop_offset < 2):
                    page_idx = page_idx0
                else:
                    page_idx = page_idx1
                scale_row = Int32(loop_offset & 1)
                # Compute word_offset in Int64: cross-layer KV layouts can
                # push ``page_idx * kscale_page_stride_words`` past int32.
                word_offset = (
                    Int64(page_idx) * Int64(kscale_page_stride_words)
                    + Int64(scale_row) * Int64(kscale_row_stride_words)
                    + Int64(head_kv) * Int64(kscale_head_kv_stride_words)
                    + Int64(lane_col)
                )
                g = cute.make_tensor(
                    kscale_words.iterator + word_offset,
                    cute.make_layout((1,)),
                )
                s = cute.make_tensor(
                    smem_sfs_k.iterator + col,
                    cute.make_layout((1,)),
                )
                cute.copy(atom, g, s)
            else:
                smem_sfs_k[col] = Float32(0.0)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

    @cute.jit
    def make_paged_kv_physical_tensors(
        self,
        kcache: cute.Tensor,
        vcache: cute.Tensor,
        h_r: Int32,
        num_head_kv: Int32,
    ):
        """Build the physical K/V TMA descriptor tensors from cache views.

        Python passes `[num_pages, block_size, Hkv, D]`.  The generated kernel's TMA
        descriptors use logical coordinates:

        * K: `[token_in_page, dim, ((head_rep, head_kv), physical_page)]`
        * V: `[dim, token_in_page, ((head_rep, head_kv), physical_page)]`

        The head-rep stride is zero, so GQA broadcasts each KV head across the
        corresponding Q-head group without materializing expanded K/V.
        """

        num_pages = kcache.shape[0]
        block_size = Int32(self.fp8_config.block_size)
        head_dim = Int32(self.fp8_config.head_dim)
        k_layout = cute.make_layout(
            (block_size, head_dim, ((h_r, num_head_kv), num_pages)),
            stride=(
                kcache.stride[1],
                kcache.stride[3],
                ((0, kcache.stride[2]), kcache.stride[0]),
            ),
        )
        v_layout = cute.make_layout(
            (head_dim, block_size, ((h_r, num_head_kv), num_pages)),
            stride=(
                vcache.stride[3],
                vcache.stride[1],
                ((0, vcache.stride[2]), vcache.stride[0]),
            ),
        )
        return (
            cute.make_tensor(kcache.iterator, k_layout),
            cute.make_tensor(vcache.iterator, v_layout),
        )

    @cute.jit
    def make_varlen_q_o_tensors(
        self,
        q: cute.Tensor,
        output: cute.Tensor,
        batch: Int32,
        max_seqlens_q: Int32,
        num_head_q: Int32,
        num_head_kv: Int32,
    ):
        """Build packed-varlen Q/O layouts over the real total-Q allocation.

        The older layout used a negative base offset plus a virtual padded batch
        mode.  Its row stride and virtual-batch stride were equal, which lets the
        TMA descriptor coalesce those modes and issue OOB Q reads on boundary
        tiles when a sequence length is not a multiple of tile_q.  Keep the TMA
        descriptor bounded by the real packed total_q rows instead; per-batch
        addressing is applied with domain_offset at the producer.
        """
        del batch, max_seqlens_q  # not used: bounds come from q/output shapes

        h_r = num_head_q // num_head_kv
        head_dim = Int32(self.fp8_config.head_dim)
        q_layout = cute.make_layout(
            (q.shape[0], head_dim, (h_r, num_head_kv)),
            stride=(
                q.stride[0],
                q.stride[2],
                (q.stride[1], q.stride[1] * h_r),
            ),
        )
        o_layout = cute.make_layout(
            (output.shape[0], head_dim, (h_r, num_head_kv)),
            stride=(
                output.stride[0],
                output.stride[2],
                (output.stride[1], output.stride[1] * h_r),
            ),
        )
        return (
            cute.make_tensor(q.iterator, q_layout),
            cute.make_tensor(output.iterator, o_layout),
        )

    @cute.jit
    def apply_sage_qk_scale(
        self,
        regs_s: cute.Tensor,
        coords_s: cute.Tensor,
        smem_sfs_q: cute.Tensor,
        smem_sfs_k: cute.Tensor,
    ):
        """Apply qscale*kscale to the QK accumulator before mask/rowmax.

        This is the logical equivalent of SageAttn::generateApplyCodeBmm1: each
        S element is scaled by its row's Q scale and column's K scale after BMM1.
        """

        for i in cutlass.range_constexpr(0, cute.size(regs_s), 2):
            row = coords_s[i][0] & Int32(127)
            col = coords_s[i][1] & Int32(127)
            row_next = coords_s[i + 1][0] & Int32(127)
            col_next = coords_s[i + 1][1] & Int32(127)
            qk_scale = cute.arch.mul_packed_f32x2(
                (smem_sfs_q[row], smem_sfs_q[row_next]),
                (smem_sfs_k[col], smem_sfs_k[col_next]),
            )
            regs_s[i], regs_s[i + 1] = cute.arch.mul_packed_f32x2(
                (regs_s[i], regs_s[i + 1]),
                qk_scale,
            )

    @cute.jit
    def apply_sage_qk_scale_and_rowmax(
        self,
        regs_s: cute.Tensor,
        coords_s: cute.Tensor,
        smem_sfs_q: cute.Tensor,
        smem_sfs_k: cute.Tensor,
        row_max: Float32,
    ):
        local_max_0 = row_max
        local_max_1 = -Float32.inf

        for i in cutlass.range_constexpr(0, cute.size(regs_s), 4):
            row = coords_s[i][0] & Int32(127)
            col = coords_s[i][1] & Int32(127)
            row_next = coords_s[i + 1][0] & Int32(127)
            col_next = coords_s[i + 1][1] & Int32(127)
            qk_scale = cute.arch.mul_packed_f32x2(
                (smem_sfs_q[row], smem_sfs_q[row_next]),
                (smem_sfs_k[col], smem_sfs_k[col_next]),
            )
            regs_s[i], regs_s[i + 1] = cute.arch.mul_packed_f32x2(
                (regs_s[i], regs_s[i + 1]),
                qk_scale,
            )
            local_max_0 = cute.arch.fmax(local_max_0, regs_s[i])
            local_max_0 = cute.arch.fmax(local_max_0, regs_s[i + 1])

            row_2 = coords_s[i + 2][0] & Int32(127)
            col_2 = coords_s[i + 2][1] & Int32(127)
            row_3 = coords_s[i + 3][0] & Int32(127)
            col_3 = coords_s[i + 3][1] & Int32(127)
            qk_scale_1 = cute.arch.mul_packed_f32x2(
                (smem_sfs_q[row_2], smem_sfs_q[row_3]),
                (smem_sfs_k[col_2], smem_sfs_k[col_3]),
            )
            regs_s[i + 2], regs_s[i + 3] = cute.arch.mul_packed_f32x2(
                (regs_s[i + 2], regs_s[i + 3]),
                qk_scale_1,
            )
            local_max_1 = cute.arch.fmax(local_max_1, regs_s[i + 2])
            local_max_1 = cute.arch.fmax(local_max_1, regs_s[i + 3])

        return cute.arch.fmax(local_max_0, local_max_1)

    @cute.jit
    def apply_fp8_p_transform(
        self,
        regs_s: cute.Tensor,
        regs_p_u32: cute.Tensor,
        row_max_safe: Float32,
        scale_softmax_log2: Float32,
        log2_p_scale: Float32,
    ):
        """Transform S to P exactly like TmemP.h for E4M3 BMM2.

        The FP32 `regs_s` values remain the denominator contribution; `regs_p_u32`
        receives the rounded E4M3 payload consumed by BMM2, packed as four bytes
        per 32-bit word to avoid the scalar-FP8 store packing path.

        ``log2_p_scale`` is the fixed per-q-head ``log2(DEFAULT_P_SCALE)``
        (= ``log2(256)``) value read from the cached p-scale tensor.
        """

        offset = (Float32(0.0) - row_max_safe) * scale_softmax_log2 + log2_p_scale
        frg_cnt = 4
        frg_tile = cute.size(regs_s) // frg_cnt
        regs_s_frg = cute.logical_divide(regs_s, cute.make_layout(frg_tile))
        regs_p_frg = cute.logical_divide(regs_p_u32, cute.make_layout(frg_tile // 4))
        for j in range(frg_cnt):
            for k in range(0, cute.size(regs_s_frg, mode=[0]), 4):
                if k == 0:
                    regs_s_frg[k, j], regs_s_frg[k + 1, j] = cute.arch.fma_packed_f32x2(
                        (regs_s_frg[k, j], regs_s_frg[k + 1, j]),
                        (scale_softmax_log2, scale_softmax_log2),
                        (offset, offset),
                    )
                else:
                    regs_s_frg[k, j], regs_s_frg[k + 1, j] = cute.arch.add_packed_f32x2(
                        cute.arch.mul_packed_f32x2(
                            (regs_s_frg[k, j], regs_s_frg[k + 1, j]),
                            (scale_softmax_log2, scale_softmax_log2),
                        ),
                        (offset, offset),
                    )
                regs_s_frg[k, j] = cute.math.exp2(regs_s_frg[k, j], fastmath=True)
                regs_s_frg[k + 1, j] = cute.math.exp2(regs_s_frg[k + 1, j], fastmath=True)
                regs_s_frg[k + 2, j], regs_s_frg[k + 3, j] = cute.arch.add_packed_f32x2(
                    cute.arch.mul_packed_f32x2(
                        (regs_s_frg[k + 2, j], regs_s_frg[k + 3, j]),
                        (scale_softmax_log2, scale_softmax_log2),
                    ),
                    (offset, offset),
                )
                regs_s_frg[k + 2, j] = cute.math.exp2(regs_s_frg[k + 2, j], fastmath=True)
                regs_s_frg[k + 3, j] = cute.math.exp2(regs_s_frg[k + 3, j], fastmath=True)
                regs_p_frg[k // 4, j] = _cvt_float4_to_e4m3_u32(
                    regs_s_frg[k, j],
                    regs_s_frg[k + 1, j],
                    regs_s_frg[k + 2, j],
                    regs_s_frg[k + 3, j],
                )

    @cute.jit
    def softmax_step_sage(
        self,
        stage: int,
        need_apply_mask: bool,
        iter_args: tuple,
        value_args: tuple,
        pipeline_args: tuple,
        atom_args: tuple,
        tensor_args: tuple,
        sage_args: tuple,
    ):
        """Softmax step with the generated SageAttn FP8-prefill semantics.

        The order intentionally follows the generated path:

        1. load S from tmem
        2. apply qscale * kscale
        3. apply bottom-right causal/residual mask
        4. compute rowmax
        5. write old/new max for correction
        6. compute exp2((S-rowmax)*log2e + log2(DEFAULT_P_SCALE) = log2(256))
        7. store E4M3 P payload while rowsum uses the unrounded FP32 value
        """

        cS, row_max, row_sum, vec_i_handle = iter_args
        (
            seqlen_k,
            seqlen_q,
            scale_softmax_log2,
            window_size_left,
            window_size_right,
            log2_p_scale,
        ) = value_args
        (
            mma_si_consumer,
            si_corr_producer,
            s0_s1_sequence_consumer,
            s0_s1_sequence_producer,
        ) = pipeline_args
        (
            qk_thr_mma,
            tiled_tmem_load,
            tiled_tmem_store,
            tiled_tmem_store_vec,
            thr_tmem_load,
            thr_tmem_store,
            thr_tmem_store_vec,
        ) = atom_args
        (
            tTMEM_LOADtS,
            tTMEM_STORE_VECtS,
            tTMEM_STOREtS_x4,
        ) = tensor_args
        smem_sfs_q, smem_sfs_k = sage_args

        tile_plike_fp32 = self.qk_mma_tiler[1] // Float32.width * self.q_dtype.width
        tScS = qk_thr_mma.partition_C(cS)
        tScS_vec_layout = cute.composition(tScS.layout, cute.make_layout((128, 2)))
        tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)

        tScS_P_layout = cute.composition(tScS.layout, cute.make_layout((128, tile_plike_fp32)))
        tScS_P = cute.make_tensor(tScS.iterator, tScS_P_layout)
        tTMEM_LOADcS = thr_tmem_load.partition_D(tScS)
        tTMEM_STORE_VECcS = thr_tmem_store_vec.partition_S(tScS_vec)
        tTMEM_STOREcS = thr_tmem_store.partition_S(tScS_P)

        si_handle = mma_si_consumer.wait_and_advance()
        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcS.shape, self.qk_acc_dtype)
        cute.copy(tiled_tmem_load, tTMEM_LOADtS, tTMEM_LOADrS)

        old_row_max = row_max
        if cutlass.const_expr(need_apply_mask):
            self.apply_sage_qk_scale(
                tTMEM_LOADrS,
                tTMEM_LOADcS,
                smem_sfs_q,
                smem_sfs_k,
            )
            fmha_utils.FusedMask.apply_mask(
                self.mask_type,
                tTMEM_LOADrS,
                tTMEM_LOADcS,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
            )
            row_max = tTMEM_LOADrS.load().reduce(cute.ReductionOp.MAX, row_max, 0)
        else:
            row_max = self.apply_sage_qk_scale_and_rowmax(
                tTMEM_LOADrS,
                tTMEM_LOADcS,
                smem_sfs_q,
                smem_sfs_k,
                row_max,
            )
        row_max_safe = row_max
        if row_max == -cutlass.Float32.inf:
            row_max_safe = 0.0
        tTMEM_STORE_VECrS = cute.make_rmem_tensor(tTMEM_STORE_VECcS.shape, self.qk_acc_dtype)
        tTMEM_STORE_VECrS[0] = old_row_max
        tTMEM_STORE_VECrS[1] = row_max_safe
        cute.copy(tiled_tmem_store_vec, tTMEM_STORE_VECrS, tTMEM_STORE_VECtS)
        vec_i_handle.commit()

        tTMEM_STORErS_x4 = cute.make_rmem_tensor(tTMEM_STOREcS.shape, self.qk_acc_dtype)
        tTMEM_STORErS_x4_u32 = cute.recast_tensor(tTMEM_STORErS_x4, Uint32)

        if cutlass.const_expr(stage == 0):
            sequence_producer_handle = s0_s1_sequence_producer.acquire_and_advance()
        else:
            sequence_consumer_handle = s0_s1_sequence_consumer.wait_and_advance()

        self.apply_fp8_p_transform(
            tTMEM_LOADrS,
            tTMEM_STORErS_x4_u32,
            row_max_safe,
            scale_softmax_log2,
            log2_p_scale,
        )

        if cutlass.const_expr(stage == 0):
            sequence_producer_handle.commit()
        else:
            sequence_consumer_handle.release()

        cute.copy(tiled_tmem_store, tTMEM_STORErS_x4, tTMEM_STOREtS_x4)
        cute.arch.fence_view_async_tmem_store()
        si_handle.release()

        vec_i_handle = si_corr_producer.acquire_and_advance()
        acc_scale_ = scale_softmax_log2 * (old_row_max - row_max_safe)
        acc_scale = cute.math.exp2(acc_scale_, fastmath=True) * 0.5
        row_sum *= acc_scale
        local_row_sum_0 = (row_sum, row_sum)
        local_row_sum_1 = (0.0, 0.0)
        local_row_sum_2 = (0.0, 0.0)
        local_row_sum_3 = (0.0, 0.0)

        reduction_unroll = 4
        frg_tile = cute.size(tTMEM_LOADrS) // reduction_unroll
        tTMEM_LOADrS_frg = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(frg_tile))

        for j in cutlass.range_constexpr(0, cute.size(tTMEM_LOADrS_frg, mode=[0]), 2):
            local_row_sum_0 = cute.arch.add_packed_f32x2(
                local_row_sum_0, (tTMEM_LOADrS_frg[j, 0], tTMEM_LOADrS_frg[j + 1, 0])
            )
            local_row_sum_1 = cute.arch.add_packed_f32x2(
                local_row_sum_1, (tTMEM_LOADrS_frg[j, 1], tTMEM_LOADrS_frg[j + 1, 1])
            )
            local_row_sum_2 = cute.arch.add_packed_f32x2(
                local_row_sum_2, (tTMEM_LOADrS_frg[j, 2], tTMEM_LOADrS_frg[j + 1, 2])
            )
            local_row_sum_3 = cute.arch.add_packed_f32x2(
                local_row_sum_3, (tTMEM_LOADrS_frg[j, 3], tTMEM_LOADrS_frg[j + 1, 3])
            )

        local_row_sum_0 = cute.arch.add_packed_f32x2(local_row_sum_0, local_row_sum_1)
        local_row_sum_2 = cute.arch.add_packed_f32x2(local_row_sum_2, local_row_sum_3)
        local_row_sum_0 = cute.arch.add_packed_f32x2(local_row_sum_0, local_row_sum_2)
        row_sum = local_row_sum_0[0] + local_row_sum_0[1]

        return (
            row_max,
            row_sum,
            vec_i_handle,
            mma_si_consumer,
            si_corr_producer,
            s0_s1_sequence_consumer,
            s0_s1_sequence_producer,
        )

    @cute.jit
    def load_sage_q_scale_tile_warpgroup(
        self,
        qscale: cute.Tensor,
        smem_sfs_q: cute.Tensor,
        batch_idx: Int32,
        head_q: Int32,
        q_tile_start: Int32,
        seqlen_q: Int32,
        thread_idx: Int32,
    ):
        row = thread_idx
        atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.ALWAYS),
            cutlass.Float32,
            num_bits_per_copy=cutlass.Float32.width,
        )
        q_local = q_tile_start + row
        if q_local < seqlen_q:
            g = cute.make_tensor(
                qscale.iterator
                + batch_idx * qscale.stride[0]
                + head_q * qscale.stride[1]
                + q_local * qscale.stride[2],
                cute.make_layout((1,)),
            )
            s = cute.make_tensor(smem_sfs_q.iterator + row, cute.make_layout((1,)))
            cute.copy(atom, g, s)
        else:
            smem_sfs_q[row] = Float32(0.0)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

    # For both softmax0 and softmax1 warp group
    @cute.jit
    def softmax_sage(
        self,
        stage: int,
        cu_seqlens_q: cute.Tensor,
        seqlens_kvcache: cute.Tensor,
        qscale: cute.Tensor,
        sSfsQ: cute.Tensor,
        sSfsK: cute.Tensor,
        scale_softmax_log2: Float32,
        qk_thr_mma: cute.ThrMma,
        tStS: cute.Tensor,
        tStSi: cute.Tensor,
        load_sage_k_consumer: pipeline.PipelineConsumer,
        mma_si_consumer: pipeline.PipelineConsumer,
        si_corr_producer: pipeline.PipelineProducer,
        s0_s1_sequence_consumer: pipeline.PipelineConsumer,
        s0_s1_sequence_producer: pipeline.PipelineProducer,
        tile_sched_params: fmha_utils.FmhaStaticTileSchedulerParams,
        p_scale_log2: cute.Tensor,
    ):
        """Compute softmax on attention scores from QK matrix multiplication.

        This method handles the softmax computation for either the first or second half of the
        attention matrix, depending on the 'stage' parameter. It calculates row-wise maximum
        and sum values needed for stable softmax computation, applies optional masking, and
        transforms raw attention scores into probability distributions.

        The implementation uses specialized memory access patterns and efficient math operations
        for computing exp(x) using exp2 functions. It also coordinates pipeline
        synchronization between MMA, correction, and sequence processing stages.

        :param stage: Processing stage (0 for first half, 1 for second half of attention matrix)
        :type stage: int
        :param seqlen_k: Length of the key sequence
        :type seqlen_k: Int32
        :param seqlen_q: Length of the query sequence
        :type seqlen_q: Int32
        :param cum_seqlen_q: Cumulative sequence lengths for queries
        :type cum_seqlen_q: cute.Tensor | None
        :param cum_seqlen_k: Cumulative sequence lengths for keys
        :type cum_seqlen_k: cute.Tensor | None
        :param scale_softmax_log2: Log2 scale factor for softmax operation
        :type scale_softmax_log2: Float32
        :param qk_thr_mma: Thread MMA operation for QK matrix multiplication
        :type qk_thr_mma: cute.ThrMma
        :param tStS: Shared tensor for softmax input/output
        :type tStS: cute.Tensor
        :param tStSi: Input tensor containing attention scores
        :type tStSi: cute.Tensor
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]
        :param mma_si_pipeline: Pipeline for synchronizing with MMA operations
        :type mma_si_pipeline: pipeline.PipelineAsync
        :param si_corr_pipeline: Pipeline for synchronizing with correction operations
        :type si_corr_pipeline: pipeline.PipelineAsync
        :param s0_s1_sequence_pipeline: Pipeline for synchronizing between stage 0 and 1
        :type s0_s1_sequence_pipeline: pipeline.PipelineAsync
        :param tile_sched_params: Parameters for tile scheduling
        :type tile_sched_params: fmha_utils.FmhaStaticTileSchedulerParams
        """
        tidx, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(stage == 0):
            softmax_threads = self.threads_per_warp * len(self.softmax0_warp_ids)
            tmem_vec_offset = self.tmem_vec0_offset
            tmem_p_offset = self.tmem_p0_offset
        else:
            softmax_threads = self.threads_per_warp * len(self.softmax1_warp_ids)
            tmem_vec_offset = self.tmem_vec1_offset
            tmem_p_offset = self.tmem_p1_offset
        thread_idx = tidx % softmax_threads

        cS_base = cute.make_identity_tensor((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
        tilePlikeFP32 = self.qk_mma_tiler[1] // 32 * self.q_dtype.width
        tScS = qk_thr_mma.partition_C(cS_base)
        tStS_vec_layout = cute.composition(tStS.layout, cute.make_layout((128, 2)))
        tStS_vec = cute.make_tensor(tStS.iterator + tmem_vec_offset, tStS_vec_layout)
        tScS_vec_layout = cute.composition(tScS.layout, cute.make_layout((128, 2)))
        tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)
        tStS_P_layout = cute.composition(tStS.layout, cute.make_layout((128, tilePlikeFP32)))
        tStS_P = cute.make_tensor(tStS.iterator + tmem_p_offset, tStS_P_layout)
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            self.qk_acc_dtype,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStSi)
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        tTMEM_LOADtS = thr_tmem_load.partition_S(tStSi)
        tmem_store_vec_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(2)),
            self.qk_acc_dtype,
        )
        tiled_tmem_store_vec = tcgen05.make_tmem_copy(tmem_store_vec_atom, tStS_vec)
        thr_tmem_store_vec = tiled_tmem_store_vec.get_slice(thread_idx)
        tTMEM_STORE_VECtS = thr_tmem_store_vec.partition_D(tStS_vec)
        tTMEM_STORE_VECcS = thr_tmem_store_vec.partition_S(tScS_vec)
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)),
            self.qk_acc_dtype,
        )
        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStS_P)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)
        tTMEM_STOREtS_x4 = thr_tmem_store.partition_D(tStS_P)

        tile_sched = fmha_utils.create_fmha_static_tile_scheduler(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        while work_tile.is_valid_tile:
            curr_block_coord = work_tile.tile_idx
            batch_coord = curr_block_coord[2][1]
            head_q = curr_block_coord[2][0]
            cuseqlen_q = cu_seqlens_q[batch_coord]
            seqlen_q_ = cu_seqlens_q[batch_coord + 1] - cuseqlen_q
            seqlen_k_ = seqlens_kvcache[batch_coord]
            continue_cond = not fmha_utils.FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                self.cta_tiler[0],
                curr_block_coord[0],
                seqlen_q_,
            )

            if not continue_cond:
                row_max = -Float32.inf
                row_sum = 0.0
                # Fixed per-q-head log2(DEFAULT_P_SCALE) = log2(256).
                log2_p_scale = p_scale_log2[head_q]
                value_args = (
                    seqlen_k_,
                    seqlen_q_,
                    scale_softmax_log2,
                    None,
                    Int32(0),
                    log2_p_scale,
                )
                atom_args = (
                    qk_thr_mma,
                    tiled_tmem_load,
                    tiled_tmem_store,
                    tiled_tmem_store_vec,
                    thr_tmem_load,
                    thr_tmem_store,
                    thr_tmem_store_vec,
                )
                tensor_args = (
                    tTMEM_LOADtS,
                    tTMEM_STORE_VECtS,
                    tTMEM_STOREtS_x4,
                )

                logical_offset = (
                    curr_block_coord[0] * self.cta_tiler[0] + stage * self.qk_mma_tiler[0],
                    0,
                )
                cS = cute.domain_offset(logical_offset, cS_base)
                vec_i_handle = si_corr_producer.acquire_and_advance()
                q_tile_start = (
                    curr_block_coord[0] * self.cta_tiler[0] + stage * self.qk_mma_tiler[0]
                )
                smem_sfs_q = sSfsQ[None, stage]
                self.load_sage_q_scale_tile_warpgroup(
                    qscale,
                    smem_sfs_q,
                    batch_coord,
                    head_q,
                    q_tile_start,
                    seqlen_q_,
                    Int32(thread_idx),
                )

                start_count = fmha_utils.FusedMask.get_trip_start(
                    self.mask_type,
                    curr_block_coord,
                    self.cta_tiler,
                    seqlen_q_,
                    seqlen_k_,
                    None,
                )

                leading_mask_count = fmha_utils.FusedMask.get_masked_leading_count(
                    self.mask_type,
                    curr_block_coord,
                    self.cta_tiler,
                    seqlen_q_,
                    seqlen_k_,
                    None,
                    Int32(0),
                )
                unmask_count = fmha_utils.FusedMask.get_unmasked_trip_count(
                    self.mask_type,
                    curr_block_coord,
                    self.cta_tiler,
                    seqlen_q_,
                    seqlen_k_,
                    None,
                    Int32(0),
                )
                trailing_mask_count = fmha_utils.FusedMask.get_masked_trailing_count(
                    self.mask_type,
                    curr_block_coord,
                    self.cta_tiler,
                    seqlen_q_,
                    seqlen_k_,
                    None,
                    Int32(0),
                )

                # Reverse KV-tile iteration (FA4-style): visit segments in
                # trailing → unmask → leading order, high→low within each
                # segment, so attention-sink tiles are folded in last.
                last_idx = start_count + leading_mask_count + unmask_count + trailing_mask_count - 1

                # Segment: trailing mask (highest KV indices, processed first)
                for j in cutlass.range(0, trailing_mask_count, 1, unroll=1):
                    i = last_idx - j
                    sfs_k_handle = load_sage_k_consumer.wait_and_advance()
                    cS_iter = cute.domain_offset((0, i * self.qk_mma_tiler[1]), cS)
                    iter_args = (cS_iter, row_max, row_sum, vec_i_handle)
                    pipeline_args = (
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    )
                    sage_args = (smem_sfs_q, sSfsK[None, sfs_k_handle.index])
                    (
                        row_max,
                        row_sum,
                        vec_i_handle,
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    ) = self.softmax_step_sage(
                        stage,
                        True,
                        iter_args,
                        value_args,
                        pipeline_args,
                        atom_args,
                        tensor_args,
                        sage_args,
                    )
                    sfs_k_handle.release()

                # Segment: unmask middle (descending)
                unmask_high = start_count + leading_mask_count + unmask_count - 1
                for j in cutlass.range(0, unmask_count, 1, unroll=1):
                    i = unmask_high - j
                    sfs_k_handle = load_sage_k_consumer.wait_and_advance()
                    cS_iter = cute.domain_offset((0, i * self.qk_mma_tiler[1]), cS)
                    iter_args = (cS_iter, row_max, row_sum, vec_i_handle)
                    pipeline_args = (
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    )
                    sage_args = (smem_sfs_q, sSfsK[None, sfs_k_handle.index])
                    (
                        row_max,
                        row_sum,
                        vec_i_handle,
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    ) = self.softmax_step_sage(
                        stage,
                        False,
                        iter_args,
                        value_args,
                        pipeline_args,
                        atom_args,
                        tensor_args,
                        sage_args,
                    )
                    sfs_k_handle.release()

                # Segment: leading mask (lowest KV indices, processed last - sink)
                leading_high = start_count + leading_mask_count - 1
                for j in cutlass.range(0, leading_mask_count, 1, unroll=1):
                    i = leading_high - j
                    sfs_k_handle = load_sage_k_consumer.wait_and_advance()
                    cS_iter = cute.domain_offset((0, i * self.qk_mma_tiler[1]), cS)
                    iter_args = (cS_iter, row_max, row_sum, vec_i_handle)
                    pipeline_args = (
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    )
                    sage_args = (smem_sfs_q, sSfsK[None, sfs_k_handle.index])
                    (
                        row_max,
                        row_sum,
                        vec_i_handle,
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    ) = self.softmax_step_sage(
                        stage,
                        True,
                        iter_args,
                        value_args,
                        pipeline_args,
                        atom_args,
                        tensor_args,
                        sage_args,
                    )
                    sfs_k_handle.release()
                si_handle = mma_si_consumer.wait_and_advance()
                tTMEM_STORE_VECrS = cute.make_rmem_tensor(
                    tTMEM_STORE_VECcS.shape, self.qk_acc_dtype
                )
                tTMEM_STORE_VECrS[0] = row_sum
                tTMEM_STORE_VECrS[1] = row_max
                cute.copy(tiled_tmem_store_vec, tTMEM_STORE_VECrS, tTMEM_STORE_VECtS)
                cute.arch.fence_view_async_tmem_store()
                vec_i_handle.commit()
                # Publish final row stats before releasing the last S tile.
                si_handle.release()

            # Advance to next tile
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()
        # End of persistent scheduler loop

    @cute.jit
    def correction_rescale(
        self,
        thr_mma: cute.ThrMma,
        tOtO: cute.Tensor,
        scale: Float32,
    ):
        """Rescale partial O in TMEM with correction warp-local indexing."""

        pv_tiled_mma_shape = (
            self.pv_mma_tiler[0],
            self.pv_mma_tiler[1],
        )
        cO = cute.make_identity_tensor(pv_tiled_mma_shape)
        tOcO = thr_mma.partition_C(cO)

        corr_tile_size = 16
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )

        tOtO_i_layout = cute.composition(tOtO.layout, cute.make_layout((128, corr_tile_size)))
        tOcO_i_layout = cute.composition(tOcO.layout, cute.make_layout((128, corr_tile_size)))

        tOtO_i = cute.make_tensor(tOtO.iterator, tOtO_i_layout)
        tOcO_i = cute.make_tensor(tOcO.iterator, tOcO_i_layout)

        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i)
        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        thread_idx = (
            warp_idx - self.correction_warp_ids[0]
        ) * self.threads_per_warp + cute.arch.lane_idx()
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)

        tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i)
        tTMEM_LOADcO = thr_tmem_load.partition_D(tOcO_i)

        tTMEM_STOREtO = thr_tmem_store.partition_D(tOtO_i)

        tTMrO = cute.make_rmem_tensor(
            (tTMEM_LOADcO.shape, 128 // corr_tile_size), self.pv_acc_dtype
        )
        for i in range(self.cta_tiler[2] // corr_tile_size):
            tTMrO_i_ = tTMrO[None, i]
            tTMrO_i_layout = cute.composition(tTMrO_i_.layout, cute.make_layout(tTMrO.shape[0]))
            tTMrO_i = cute.make_tensor(tTMrO_i_.iterator, tTMrO_i_layout)
            tTMEM_LOADtO_i = cute.make_tensor(
                tTMEM_LOADtO.iterator + i * corr_tile_size, tTMEM_LOADtO.layout
            )
            tTMEM_STOREtO_i = cute.make_tensor(
                tTMEM_STOREtO.iterator + i * corr_tile_size, tTMEM_STOREtO.layout
            )

            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO_i)
            for j in range(0, cute.size(tTMrO_i), 2):
                tTMrO_i[j], tTMrO_i[j + 1] = cute.arch.mul_packed_f32x2(
                    (tTMrO_i[j], tTMrO_i[j + 1]),
                    (scale, scale),
                )
            cute.copy(tiled_tmem_store, tTMrO_i, tTMEM_STOREtO_i)

    @cute.jit
    def correction_epilog(
        self,
        thr_mma: cute.ThrMma,
        tOtO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        tTMEM_LOAD_VECrS: cute.Tensor,
        row_idx: Int32,
        cuseqlen_q: Int32,
        seqlen_q: Int32,
        head_q: Int32,
        blk_coord: Int32,
        scale_softmax: Float32,
        scale: Float32,
        output: cute.Tensor,
    ):
        pv_tiled_mma_shape = (
            self.pv_mma_tiler[0],
            self.pv_mma_tiler[1],
        )
        cO = cute.make_identity_tensor(pv_tiled_mma_shape)

        corr_tile_size = 32 * 8 // self.o_dtype.width
        tOcO = thr_mma.partition_C(cO)

        tOtO_i = cute.logical_divide(tOtO, cute.make_layout((128, corr_tile_size)))
        tOcO_i = cute.logical_divide(tOcO, cute.make_layout((128, corr_tile_size)))
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        thread_idx = (
            warp_idx - self.correction_warp_ids[0]
        ) * self.threads_per_warp + cute.arch.lane_idx()

        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils.get_tmem_load_op(
            self.pv_mma_tiler,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=False,
        )

        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_i[(None, None), 0])

        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)

        tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tTMEM_LOADoO = thr_tmem_load.partition_D(tOcO_i[(None, None), None])

        for i in range(self.cta_tiler[2] // corr_tile_size):
            tTMEM_LOADtO_i = tTMEM_LOADtO[None, 0, 0, i]
            tTMEM_LOADoO_i = tTMEM_LOADoO[None, 0, 0, i]
            tTMrO = cute.make_rmem_tensor(tTMEM_LOADoO_i.shape, self.pv_acc_dtype)
            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO)
            for j in range(0, cute.size(tTMrO), 2):
                tTMrO[j], tTMrO[j + 1] = cute.arch.mul_packed_f32x2(
                    (tTMrO[j], tTMrO[j + 1]),
                    (scale, scale),
                )
            tSMrO = cute.make_rmem_tensor(tTMrO.shape, self.o_dtype)
            o_vec = tTMrO.load()
            tSMrO.store(o_vec.to(self.o_dtype))

            if row_idx < seqlen_q:
                col = tTMEM_LOADoO_i[0][1]
                out_offset = (
                    (cuseqlen_q + row_idx) * output.stride[0] + head_q * output.stride[1] + col
                )
                out_ptr = cute.make_ptr(
                    self.o_dtype,
                    (output.iterator + out_offset).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                tGOrO = cute.make_tensor(out_ptr, tSMrO.layout)
                cute.autovec_copy(tSMrO, tGOrO)

        if cutlass.const_expr(mLSE is not None):
            scaled_tmp = scale_softmax * tTMEM_LOAD_VECrS[1]
            lse = cute.math.log(tTMEM_LOAD_VECrS[0], fastmath=True) + scaled_tmp
            if row_idx < seqlen_q:
                mLSE[row_idx + cuseqlen_q, blk_coord[2]] = lse

    #  GPU device kernel
    @cute.kernel
    def kernel_sage_paged(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        qk_page_tiled_mma: cute.TiledMma,
        pv_page_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_k_page: cute.CopyAtom,
        mK_pdl: cute.Tensor,
        tma_atom_v_page: cute.CopyAtom,
        mV_pdl: cute.Tensor,
        output: cute.Tensor,
        qscale: cute.Tensor,
        kscale_words: cute.Tensor,
        vscale: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        block_ids: cute.Tensor,
        seqlens_kvcache: cute.Tensor,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        k_page_smem_layout_staged: cute.ComposedLayout,
        p_tmem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        v_page_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: fmha_utils.FmhaStaticTileSchedulerParams,
        h_r: Int32,
        num_head_kv: Int32,
        kscale_page_stride_words: Int32,
        kscale_row_stride_words: Int32,
        kscale_head_kv_stride_words: Int32,
        p_scale_log2: cute.Tensor,
    ):
        """The device kernel implementation of the Fused Multi-Head Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the FMHA computation:
        1. Load warp: Loads Q, K, V data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Softmax warps: Compute softmax normalization on attention scores
        4. Correction warps: Apply adjustments and directly store output

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases, and optional attention masking.

        :param qk_tiled_mma: Tiled MMA for Q*K^T
        :type qk_tiled_mma: cute.TiledMma
        :param pv_tiled_mma: Tiled MMA for P*V
        :type pv_tiled_mma: cute.TiledMma
        :param tma_atom_q: TMA copy atom for query tensor
        :type tma_atom_q: cute.CopyAtom
        :param mQ_qdl: Partitioned query tensor
        :type mQ_qdl: cute.Tensor
        :param tma_atom_k: TMA copy atom for key tensor
        :type tma_atom_k: cute.CopyAtom
        :param mK_kdl: Partitioned key tensor
        :type mK_kdl: cute.Tensor
        :param tma_atom_v: TMA copy atom for value tensor
        :type tma_atom_v: cute.CopyAtom
        :param mV_dkl: Partitioned value tensor
        :type mV_dkl: cute.Tensor
        :param scale_softmax_log2: The log2 scale factor for softmax
        :type scale_softmax_log2: Float32
        :param scale_output: The scale factor for the output
        :type scale_output: Float32
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]
        :param q_smem_layout_staged: Shared memory layout for query tensor
        :type q_smem_layout_staged: cute.ComposedLayout
        :param k_smem_layout_staged: Shared memory layout for key tensor
        :type k_smem_layout_staged: cute.ComposedLayout
        :param p_tmem_layout_staged: Tensor memory layout for probability matrix
        :type p_tmem_layout_staged: cute.ComposedLayout
        :param v_smem_layout_staged: Shared memory layout for value tensor
        :type v_smem_layout_staged: cute.ComposedLayout
        :param tile_sched_params: Scheduling parameters for work distribution
        :type tile_sched_params: fmha_utils.FmhaStaticTileSchedulerParams
        """
        scale_softmax_log2 = Float32(SCALE_SOFTMAX_LOG2_HEAD_DIM_128)
        scale_softmax = Float32(INV_SQRT_HEAD_DIM_128)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Prefetch tma desc
        #
        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k_page)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v_page)

        # Alloc
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.q_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
        ).make_participants()
        load_kv_producer, load_kv_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.kv_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_kv_bytes,
            barrier_storage=storage.load_kv_mbar_ptr.data_ptr(),
        ).make_participants()
        load_sage_k_producer, load_sage_k_consumer = pipeline.PipelineAsync.create(
            num_stages=self.sage_k_stage,
            producer_group=make_thread_cooperative_group(self.threads_per_warp),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len((*self.softmax0_warp_ids, *self.softmax1_warp_ids))
            ),
            barrier_storage=storage.load_sage_k_mbar_ptr.data_ptr(),
        ).make_participants()
        mma_s0_producer, mma_s0_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_softmax_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax0_warp_ids)
            ),
            barrier_storage=storage.mma_s0_mbar_ptr.data_ptr(),
        ).make_participants()
        mma_s1_producer, mma_s1_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_softmax_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax1_warp_ids)
            ),
            barrier_storage=storage.mma_s1_mbar_ptr.data_ptr(),
        ).make_participants()
        s0_corr_producer, s0_corr_consumer = pipeline.PipelineAsync.create(
            num_stages=self.softmax_corr_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax0_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.s0_corr_mbar_ptr.data_ptr(),
        ).make_participants()
        s1_corr_producer, s1_corr_consumer = pipeline.PipelineAsync.create(
            num_stages=self.softmax_corr_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax1_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.s1_corr_mbar_ptr.data_ptr(),
        ).make_participants()
        mma_corr_producer, mma_corr_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_corr_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.mma_corr_mbar_ptr.data_ptr(),
        ).make_participants()
        s0_s1_sequence_producer, s0_s1_sequence_consumer = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax0_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax1_warp_ids)
            ),
            barrier_storage=storage.s0_s1_sequence_mbar_ptr.data_ptr(),
        ).make_participants()
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.data_ptr()

        #  Correction & Epilogue & tmem barrier init
        if warp_idx == self.empty_warp_id:
            cute.arch.mbarrier_init(
                tmem_dealloc_mbar_ptr,
                self.threads_per_warp
                * len(
                    (
                        *self.softmax0_warp_ids,
                        *self.softmax1_warp_ids,
                        *self.correction_warp_ids,
                    )
                ),
            )
        cute.arch.mbarrier_init_fence()

        #  Generate smem tensor Q/K/V
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        # Strip swizzle info to reuse smem
        sV_ptr = cute.recast_ptr(sK.iterator, v_smem_layout_staged.inner)
        sV = cute.make_tensor(sV_ptr, v_smem_layout_staged.outer)
        sSfsQ = storage.sSfsQ.get_tensor(
            cute.make_layout((self.fp8_config.tile_q, self.q_stage), stride=(1, 128))
        )
        sSfsK = storage.sSfsK.get_tensor(
            cute.make_layout((self.fp8_config.tile_kv, self.sage_k_stage), stride=(1, 128))
        )
        qk_thr_mma = qk_tiled_mma.get_slice(0)  # default 1sm
        pv_thr_mma = pv_tiled_mma.get_slice(0)  # default 1sm
        tSrQ = qk_thr_mma.make_fragment_A(sQ)
        tSrK = qk_thr_mma.make_fragment_B(sK)
        tOrV = pv_thr_mma.make_fragment_B(sV)
        qk_acc_shape = qk_thr_mma.partition_shape_C((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
        tStS = qk_thr_mma.make_fragment_C(qk_acc_shape)
        pv_acc_shape = pv_thr_mma.partition_shape_C((self.pv_mma_tiler[0], self.pv_mma_tiler[1]))
        tOtO = pv_thr_mma.make_fragment_C(pv_acc_shape)

        tStS0 = cute.make_tensor(tStS.iterator + self.tmem_s0_offset, tStS.layout)
        tStS1 = cute.make_tensor(tStS.iterator + self.tmem_s1_offset, tStS.layout)
        tOtO0 = cute.make_tensor(tOtO.iterator + self.tmem_o0_offset, tOtO.layout)
        tOtO1 = cute.make_tensor(tOtO.iterator + self.tmem_o1_offset, tOtO.layout)

        tP = cute.make_tensor(tStS.iterator, p_tmem_layout_staged.outer)
        tOrP = pv_thr_mma.make_fragment_A(tP)[None, None, None, 0]
        tOrP0 = cute.make_tensor(
            tOrP.iterator + self.qk_acc_dtype.width // self.q_dtype.width * self.tmem_p0_offset,
            tOrP.layout,
        )
        tOrP1 = cute.make_tensor(
            tOrP.iterator + self.qk_acc_dtype.width // self.q_dtype.width * self.tmem_p1_offset,
            tOrP.layout,
        )
        self.cta_sync_barrier.arrive_and_wait()
        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.empty_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

            tile_sched = fmha_utils.create_fmha_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                batch_coord = curr_block_coord[2][1]
                cuseqlen_q = cu_seqlens_q[batch_coord]
                seqlen_q = cu_seqlens_q[batch_coord + 1] - cuseqlen_q
                continue_cond = (
                    not fmha_utils.FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.cta_tiler[0],
                        curr_block_coord[0],
                        seqlen_q,
                    )
                )
                if not continue_cond:
                    mQ_qdl_ = mQ_qdl
                    seqlen_k = seqlens_kvcache[batch_coord]
                    curr_block_coord_q = curr_block_coord
                    head_q = curr_block_coord[2][0]
                    head_kv = head_q // h_r

                    logical_offset_mQ = (cuseqlen_q, 0, (0, 0))
                    mQ_qdl_ = cute.domain_offset(logical_offset_mQ, mQ_qdl)
                    curr_block_coord_q = (
                        curr_block_coord[0],
                        curr_block_coord[1],
                        head_q,
                    )

                    # Local tile partition global tensors
                    # (bM, bK, loopM, loopK, loopL)
                    gQ_qdl = cute.flat_divide(mQ_qdl_, cute.select(self.qk_mma_tiler, mode=[0, 2]))
                    tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
                    tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_q,
                        0,  # no multicast
                        cute.make_layout(1),
                        cute.group_modes(sQ, 0, 3),
                        cute.group_modes(tSgQ_qdl, 0, 3),
                    )
                    tQgQ = tQgQ_qdl[None, None, 0, curr_block_coord_q[2]]

                    # Q0
                    q0_coord = 2 * curr_block_coord_q[0]
                    q0_handle = load_q_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_q,
                        tQgQ[None, q0_coord],
                        tQsQ[None, q0_handle.index],
                        tma_bar_ptr=q0_handle.barrier,
                    )
                    # K_last (FA4-style reverse KV-tile iteration).
                    seqlen_kv_loop_start = fmha_utils.FusedMask.get_trip_start(
                        self.mask_type,
                        curr_block_coord,
                        self.cta_tiler,
                        seqlen_q,
                        seqlen_k,
                        None,
                    )
                    seqlen_kv_loop_steps = (
                        fmha_utils.FusedMask.get_trip_count(
                            self.mask_type,
                            curr_block_coord,
                            self.cta_tiler,
                            seqlen_q,
                            seqlen_k,
                            None,
                            Int32(0),
                        )
                        - 1
                    )
                    # Start from the last KV tile; loop will decrement.
                    kv_coord = seqlen_kv_loop_start + seqlen_kv_loop_steps
                    kv_tile_start = kv_coord * Int32(self.fp8_config.tile_kv)
                    page_idx0, page_idx1 = self.load_kv_page_indices(
                        block_ids,
                        batch_coord,
                        kv_tile_start,
                        seqlen_k,
                    )
                    k_handle = load_kv_producer.acquire_and_advance()
                    self.copy_paged_k_tile_tma(
                        qk_page_tiled_mma,
                        tma_atom_k_page,
                        mK_pdl,
                        sK,
                        head_kv,
                        page_idx0,
                        page_idx1,
                        k_page_smem_layout_staged,
                        k_handle.index,
                        k_handle.barrier,
                    )
                    sfs_k_handle = load_sage_k_producer.acquire_and_advance()
                    self.load_sage_k_scale_tile(
                        kscale_words,
                        sSfsK[None, sfs_k_handle.index],
                        head_kv,
                        page_idx0,
                        page_idx1,
                        kv_tile_start,
                        seqlen_k,
                        num_head_kv,
                        kscale_page_stride_words,
                        kscale_row_stride_words,
                        kscale_head_kv_stride_words,
                    )
                    sfs_k_handle.commit()
                    # Q1
                    q1_coord = q0_coord + 1
                    q1_load_coord = q1_coord
                    if q1_coord * Int32(self.qk_mma_tiler[0]) >= seqlen_q:
                        q1_load_coord = q0_coord
                    q1_handle = load_q_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_q,
                        tQgQ[None, q1_load_coord],
                        tQsQ[None, q1_handle.index],
                        tma_bar_ptr=q1_handle.barrier,
                    )
                    # V_last (paired with K_last loaded above)
                    v_handle = load_kv_producer.acquire_and_advance()
                    self.copy_paged_v_tile_tma(
                        pv_page_tiled_mma,
                        tma_atom_v_page,
                        mV_pdl,
                        sV,
                        head_kv,
                        page_idx0,
                        page_idx1,
                        v_page_smem_layout_staged,
                        v_handle.index,
                        v_handle.barrier,
                    )
                    kv_coord -= 1

                    for i in cutlass.range(0, seqlen_kv_loop_steps, 1, unroll=1):
                        # K_{last-1-i}
                        kv_tile_start = kv_coord * Int32(self.fp8_config.tile_kv)
                        page_idx0, page_idx1 = self.load_kv_page_indices(
                            block_ids,
                            batch_coord,
                            kv_tile_start,
                            seqlen_k,
                        )
                        k_handle = load_kv_producer.acquire_and_advance()
                        self.copy_paged_k_tile_tma(
                            qk_page_tiled_mma,
                            tma_atom_k_page,
                            mK_pdl,
                            sK,
                            head_kv,
                            page_idx0,
                            page_idx1,
                            k_page_smem_layout_staged,
                            k_handle.index,
                            k_handle.barrier,
                        )
                        sfs_k_handle = load_sage_k_producer.acquire_and_advance()
                        self.load_sage_k_scale_tile(
                            kscale_words,
                            sSfsK[None, sfs_k_handle.index],
                            head_kv,
                            page_idx0,
                            page_idx1,
                            kv_tile_start,
                            seqlen_k,
                            num_head_kv,
                            kscale_page_stride_words,
                            kscale_row_stride_words,
                            kscale_head_kv_stride_words,
                        )
                        sfs_k_handle.commit()
                        # V_i
                        v_handle = load_kv_producer.acquire_and_advance()
                        self.copy_paged_v_tile_tma(
                            pv_page_tiled_mma,
                            tma_atom_v_page,
                            mV_pdl,
                            sV,
                            head_kv,
                            page_idx0,
                            page_idx1,
                            v_page_smem_layout_staged,
                            v_handle.index,
                            v_handle.barrier,
                        )
                        kv_coord -= 1
                    # End of seqlen_kv loop

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                # End of persistent scheduler loop

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

            # Alloc tmem buffer
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            self.tmem_alloc_barrier.arrive_and_wait()
            tile_sched = fmha_utils.create_fmha_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                batch_coord = curr_block_coord[2][1]
                cuseqlen_q = cu_seqlens_q[batch_coord]
                seqlen_q = cu_seqlens_q[batch_coord + 1] - cuseqlen_q
                continue_cond = (
                    not fmha_utils.FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.cta_tiler[0],
                        curr_block_coord[0],
                        seqlen_q,
                    )
                )

                if not continue_cond:
                    seqlen_k = seqlens_kvcache[batch_coord]

                    # GEMM_QK00 (Q0 * K0 -> S0)
                    # 1. wait for Q0
                    q0_handle = load_q_consumer.wait_and_advance()
                    tSrQ0 = tSrQ[None, None, None, q0_handle.index]
                    # 2. wait for K0
                    k_handle = load_kv_consumer.wait_and_advance()
                    tSrK0 = tSrK[None, None, None, k_handle.index]
                    # 3. acquire empty S0 buffer
                    s0_handle = mma_s0_producer.acquire_and_advance()
                    # 4. gemm
                    num_kphases = cute.size(tSrQ0, mode=[2])
                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                        kphase_coord = (None, None, kphase_idx)
                        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                        cute.gemm(
                            qk_tiled_mma,
                            tStS0,
                            tSrQ0[kphase_coord],
                            tSrK0[kphase_coord],
                            tStS0,
                        )
                    # 5. release S0
                    s0_handle.commit()
                    # End of GEMM (Q0 * K0 -> S0)

                    # GEMM_QK10 (Q1 * K0 -> S1), K0 is ready in GEMM_QK00
                    # 1. wait for Q1
                    q1_handle = load_q_consumer.wait_and_advance()
                    tSrQ1 = tSrQ[None, None, None, q1_handle.index]
                    # 2. acquire empty S1
                    s1_handle = mma_s1_producer.acquire_and_advance()
                    # 3. gemm
                    num_kphases = cute.size(tSrQ1, mode=[2])
                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                        kphase_coord = (None, None, kphase_idx)
                        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                        cute.gemm(
                            qk_tiled_mma,
                            tStS1,
                            tSrQ1[kphase_coord],
                            tSrK0[kphase_coord],
                            tStS1,
                        )
                    # 4. release K0, then publish S1
                    k_handle.release()
                    s1_handle.commit()
                    # End of GEMM (Q1 * K0 -> S1)
                    # Note: Q0 & Q1 are still needed in the seqlen_kv loop
                    # so we need to release them after the seqlen_kv loop

                    # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                    # 1. wait for V0
                    v_handle = load_kv_consumer.wait_and_advance()
                    tOrVi = tOrV[None, None, None, v_handle.index]
                    # 2. acquire corrected O0_partial
                    # Note: acquire corr first to take it out of the critical
                    # path since softmax takes longer
                    o0_handle = mma_corr_producer.acquire_and_advance()
                    # 3. acquire P0
                    # this acquire returns the ownership of all of S0 to the mma warp
                    # including the P0 part (inplaced in S0)
                    s0_handle = mma_s0_producer.acquire_and_advance()
                    # 4. gemm
                    num_kphases = cute.size(tOrP0, mode=[2])
                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                        kphase_coord = (None, None, kphase_idx)
                        pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                        cute.gemm(
                            pv_tiled_mma,
                            tOtO0,
                            tOrP0[kphase_coord],
                            tOrVi[kphase_coord],
                            tOtO0,
                        )
                    # 5. release accumulated O0_partial
                    o0_handle.commit()
                    # End of GEMM_PV00 (P0 * V0 -> O0_partial)

                    seqlen_kv_loop_steps = (
                        fmha_utils.FusedMask.get_trip_count(
                            self.mask_type,
                            curr_block_coord,
                            self.cta_tiler,
                            seqlen_q,
                            seqlen_k,
                            None,
                            Int32(0),
                        )
                        - 1
                    )

                    # O1 hasn't been accumulated yet, its first MMA calculation doesn't need to accumulate
                    pv_whether_acc = False
                    for i in cutlass.range(0, seqlen_kv_loop_steps, 1, unroll=1):
                        # GEMM_QK0i (Q0 * Ki -> S0)
                        # 1. wait for Ki
                        k_handle = load_kv_consumer.wait_and_advance()
                        tSrKi = tSrK[None, None, None, k_handle.index]
                        # 2. gemm
                        inner_num_kphases = cute.size(tSrQ0, mode=[2])
                        for kphase_idx in cutlass.range(inner_num_kphases, unroll_full=True):
                            kphase_coord = (None, None, kphase_idx)
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                            cute.gemm(
                                qk_tiled_mma,
                                tStS0,
                                tSrQ0[kphase_coord],
                                tSrKi[kphase_coord],
                                tStS0,
                            )
                        # 3. release S0
                        s0_handle.commit()
                        # End of GEMM_QK0i (Q0 * Ki -> S0)

                        # GEMM_PV1(i-1) (P1 * V(i-1) -> O1_partial), V(i-1) is ready in GEMM_PV0(i-1)
                        # 1. acquire corrected O1_partial
                        o1_handle = mma_corr_producer.acquire_and_advance()
                        # 2. acquire P1
                        s1_handle = mma_s1_producer.acquire_and_advance()
                        # 3. gemm
                        inner_num_kphases = cute.size(tOrP0, mode=[2])
                        for kphase_idx in cutlass.range(inner_num_kphases, unroll_full=True):
                            kphase_coord = (None, None, kphase_idx)
                            pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
                            cute.gemm(
                                pv_tiled_mma,
                                tOtO1,
                                tOrP1[kphase_coord],
                                tOrVi[kphase_coord],
                                tOtO1,
                            )
                            pv_whether_acc = True
                        # 4. release accumulated O1_partial
                        o1_handle.commit()
                        # 5. release V(i-1)
                        v_handle.release()
                        # End of GEMM_PV1(i-1) (P1 * V(i-1) -> O1_partial)

                        # GEMM_QK1i (Q1 * Ki -> S1), Q1 is ready in GEMM_QK10; Ki is ready in GEMM_QK0i
                        # 1. gemm
                        inner_num_kphases = cute.size(tSrQ1, mode=[2])
                        for kphase_idx in cutlass.range(inner_num_kphases, unroll_full=True):
                            kphase_coord = (None, None, kphase_idx)
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                            cute.gemm(
                                qk_tiled_mma,
                                tStS1,
                                tSrQ1[kphase_coord],
                                tSrKi[kphase_coord],
                                tStS1,
                            )
                        # 2. release Ki, then publish S1
                        k_handle.release()
                        s1_handle.commit()
                        # End of GEMM_QK1i (Q1 * Ki -> S1)

                        # GEMM_PV0i (P0 * Vi -> O0_partial)
                        # 1. wait for Vi
                        v_handle = load_kv_consumer.wait_and_advance()
                        tOrVi = tOrV[None, None, None, v_handle.index]
                        # 2. acquire corrected O0_partial
                        o0_handle = mma_corr_producer.acquire_and_advance()
                        # 3. acquire P0
                        s0_handle = mma_s0_producer.acquire_and_advance()
                        # 4. gemm
                        inner_num_kphases = cute.size(tOrP0, mode=[2])
                        for kphase_idx in cutlass.range(inner_num_kphases, unroll_full=True):
                            kphase_coord = (None, None, kphase_idx)
                            pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            cute.gemm(
                                pv_tiled_mma,
                                tOtO0,
                                tOrP0[kphase_coord],
                                tOrVi[kphase_coord],
                                tOtO0,
                            )
                        # 5. release accumulated O0_partial
                        o0_handle.commit()
                        # End of GEMM_PV0i (P0 * Vi -> O0_partial)
                    # End of seqlen_kv loop

                    # release Q0 & Q1
                    q0_handle.release()
                    q1_handle.release()

                    # GEMM_PV1(i_end) (P1 * Vi_end -> O1)
                    # 1. acquire corrected O1_partial
                    o1_handle = mma_corr_producer.acquire_and_advance()
                    # 2. acquire P1
                    s1_handle = mma_s1_producer.acquire_and_advance()
                    # 3. gemm
                    num_kphases = cute.size(tOrP1, mode=[2])
                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                        kphase_coord = (None, None, kphase_idx)
                        pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
                        cute.gemm(
                            pv_tiled_mma,
                            tOtO1,
                            tOrP1[kphase_coord],
                            tOrVi[kphase_coord],
                            tOtO1,
                        )
                        pv_whether_acc = True
                    # 4. commit accumulated O1
                    o1_handle.commit()
                    # 5. release Vi_end
                    v_handle.release()
                    # End of GEMM_PV1(i_end) (P1 * Vi_end -> O1)

                    # Commit S0 and S1
                    s0_handle.commit()
                    s1_handle.commit()

                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            # End of persistent scheduler loop

            # dealloc tmem buffer
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            #  Retrieving tmem ptr and make acc
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                Float32,
                alignment=16,
                ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
            )
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue (direct-store path)
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.epilogue_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax0
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx < self.softmax1_warp_ids[0]:
            # increase register after decreasing
            cute.arch.warpgroup_reg_alloc(self.num_regs_softmax)

            self.softmax_sage(
                stage=0,
                cu_seqlens_q=cu_seqlens_q,
                seqlens_kvcache=seqlens_kvcache,
                qscale=qscale,
                sSfsQ=sSfsQ,
                sSfsK=sSfsK,
                scale_softmax_log2=scale_softmax_log2,
                qk_thr_mma=qk_thr_mma,
                tStS=tStS,
                tStSi=tStS0,
                load_sage_k_consumer=load_sage_k_consumer,
                mma_si_consumer=mma_s0_consumer,
                si_corr_producer=s0_corr_producer,
                s0_s1_sequence_consumer=s0_s1_sequence_consumer,
                s0_s1_sequence_producer=s0_s1_sequence_producer,
                tile_sched_params=tile_sched_params,
                p_scale_log2=p_scale_log2,
            )
            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax1
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx < self.correction_warp_ids[0] and warp_idx >= self.softmax1_warp_ids[0]:
            # increase register after decreasing
            cute.arch.warpgroup_reg_alloc(self.num_regs_softmax)

            self.softmax_sage(
                stage=1,
                cu_seqlens_q=cu_seqlens_q,
                seqlens_kvcache=seqlens_kvcache,
                qscale=qscale,
                sSfsQ=sSfsQ,
                sSfsK=sSfsK,
                scale_softmax_log2=scale_softmax_log2,
                qk_thr_mma=qk_thr_mma,
                tStS=tStS,
                tStSi=tStS1,
                load_sage_k_consumer=load_sage_k_consumer,
                mma_si_consumer=mma_s1_consumer,
                si_corr_producer=s1_corr_producer,
                s0_s1_sequence_consumer=s0_s1_sequence_consumer,
                s0_s1_sequence_producer=s0_s1_sequence_producer,
                tile_sched_params=tile_sched_params,
                p_scale_log2=p_scale_log2,
            )
            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_correction)

            cS = cute.make_identity_tensor((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
            tScS = qk_thr_mma.partition_C(cS)

            tStS_vec_layout = cute.composition(tStS.layout, cute.make_layout((128, 2)))

            tStS_vec0 = cute.make_tensor(tStS.iterator + self.tmem_vec0_offset, tStS_vec_layout)
            tStS_vec1 = cute.make_tensor(tStS.iterator + self.tmem_vec1_offset, tStS_vec_layout)

            tScS_vec_layout = cute.composition(tScS.layout, cute.make_layout((128, 2)))
            tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)

            tmem_load_v_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(2)),
                self.qk_acc_dtype,
            )

            tiled_tmem_load_vec = tcgen05.make_tmem_copy(tmem_load_v_atom, tStS_vec0)
            thread_idx = (
                warp_idx - self.correction_warp_ids[0]
            ) * self.threads_per_warp + cute.arch.lane_idx()
            thr_tmem_load_vec = tiled_tmem_load_vec.get_slice(thread_idx)

            tTMEM_LOAD_VECtS0 = thr_tmem_load_vec.partition_S(tStS_vec0)
            tTMEM_LOAD_VECtS1 = thr_tmem_load_vec.partition_S(tStS_vec1)
            tTMEM_LOAD_VECcS = thr_tmem_load_vec.partition_D(tScS_vec)

            tile_sched = fmha_utils.create_fmha_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                curr_block_coord_lse = curr_block_coord
                batch_coord = curr_block_coord[2][1]
                seqlen_k = seqlens_kvcache[batch_coord]
                cuseqlen_q = cu_seqlens_q[batch_coord]
                seqlen_q = cu_seqlens_q[batch_coord + 1] - cuseqlen_q
                curr_block_coord_lse = (
                    curr_block_coord[0],
                    curr_block_coord[1],
                    (curr_block_coord[2][0], 0),
                )
                continue_cond = (
                    not fmha_utils.FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.cta_tiler[0],
                        curr_block_coord[0],
                        seqlen_q,
                    )
                )

                if not continue_cond:
                    row_idx = curr_block_coord[0] * self.cta_tiler[0] + tTMEM_LOAD_VECcS[0][0]
                    head_q = curr_block_coord[2][0]
                    head_kv = curr_block_coord[2][0] // h_r
                    v_scale = vscale[head_kv]
                    # Ignore first signal from softmax as no correction is required
                    vec0_handle = s0_corr_consumer.wait_and_advance()
                    vec0_handle.release()
                    vec1_handle = s1_corr_consumer.wait_and_advance()

                    seqlen_kv_loop_steps = (
                        fmha_utils.FusedMask.get_trip_count(
                            self.mask_type,
                            curr_block_coord,
                            self.cta_tiler,
                            seqlen_q,
                            seqlen_k,
                            None,
                            Int32(0),
                        )
                        - 1
                    )
                    for i in cutlass.range(0, seqlen_kv_loop_steps, 1, unroll=1):
                        # wait for vec0 (row_wise current max & previous max)
                        vec0_handle = s0_corr_consumer.wait_and_advance()
                        tTMEM_LOAD_VECrS = cute.make_rmem_tensor(
                            tTMEM_LOAD_VECcS.shape, self.qk_acc_dtype
                        )
                        cute.copy(tiled_tmem_load_vec, tTMEM_LOAD_VECtS0, tTMEM_LOAD_VECrS)
                        scale_ = scale_softmax_log2 * (tTMEM_LOAD_VECrS[0] - tTMEM_LOAD_VECrS[1])
                        scale = cute.math.exp2(scale_, fastmath=True)
                        # wait for o0
                        o0_handle = mma_corr_consumer.wait_and_advance()
                        self.correction_rescale(pv_thr_mma, tOtO0, scale)
                        # release vec1 & o0
                        vec1_handle.release()
                        cute.arch.fence_view_async_tmem_store()
                        o0_handle.release()

                        # wait for vec1 (row_wise current max & previous max)
                        vec1_handle = s1_corr_consumer.wait_and_advance()
                        cute.copy(tiled_tmem_load_vec, tTMEM_LOAD_VECtS1, tTMEM_LOAD_VECrS)
                        scale_ = scale_softmax_log2 * (tTMEM_LOAD_VECrS[0] - tTMEM_LOAD_VECrS[1])
                        scale = cute.math.exp2(scale_, fastmath=True)
                        o1_handle = mma_corr_consumer.wait_and_advance()
                        self.correction_rescale(pv_thr_mma, tOtO1, scale)
                        vec0_handle.release()
                        cute.arch.fence_view_async_tmem_store()
                        o1_handle.release()
                    # End of seqlen_corr_loop_steps
                    vec1_handle.release()

                    # wait for vec0 (row_wise global sum)
                    vec0_handle = s0_corr_consumer.wait_and_advance()
                    tTMEM_LOAD_VECrS = cute.make_rmem_tensor(
                        tTMEM_LOAD_VECcS.shape, self.qk_acc_dtype
                    )
                    cute.copy(tiled_tmem_load_vec, tTMEM_LOAD_VECtS0, tTMEM_LOAD_VECrS)
                    cute.arch.fence_view_async_tmem_load()
                    vec0_handle.release()
                    # wait for o0
                    o0_handle = mma_corr_consumer.wait_and_advance()
                    self.correction_epilog(
                        pv_thr_mma,
                        tOtO0,
                        None,
                        tTMEM_LOAD_VECrS,
                        row_idx,
                        cuseqlen_q,
                        seqlen_q,
                        head_q,
                        curr_block_coord_lse,
                        scale_softmax,
                        v_scale / tTMEM_LOAD_VECrS[0],
                        output,
                    )
                    o0_handle.release()

                    # wait for vec1 (row_wise global sum)
                    vec1_handle = s1_corr_consumer.wait_and_advance()
                    cute.copy(tiled_tmem_load_vec, tTMEM_LOAD_VECtS1, tTMEM_LOAD_VECrS)
                    cute.arch.fence_view_async_tmem_load()
                    vec1_handle.release()
                    # wait for o1
                    o1_handle = mma_corr_consumer.wait_and_advance()
                    row_idx += self.qk_mma_tiler[0]
                    self.correction_epilog(
                        pv_thr_mma,
                        tOtO1,
                        None,
                        tTMEM_LOAD_VECrS,
                        row_idx,
                        cuseqlen_q,
                        seqlen_q,
                        head_q,
                        curr_block_coord_lse,
                        scale_softmax,
                        v_scale / tTMEM_LOAD_VECrS[0],
                        output,
                    )
                    o1_handle.release()
                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            # End of persistent scheduler loop
            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)
        return


def _validate_inputs(
    q: torch.Tensor,
    kcache: torch.Tensor,
    vcache: torch.Tensor,
    qscale: torch.Tensor,
    kscale: torch.Tensor,
    vscale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_ids: torch.Tensor,
    seqlens_kvcache: torch.Tensor,
    output: Optional[torch.Tensor],
    config: Fp8PagedPrefillConfig,
) -> Tuple[int, int, int, int, int]:
    tensors = (q, kcache, vcache, qscale, kscale, vscale, cu_seqlens_q, block_ids, seqlens_kvcache)
    if not all(t.is_cuda for t in tensors):
        raise ValueError("all inputs must be CUDA tensors")
    if q.dtype != torch.float8_e4m3fn or kcache.dtype != q.dtype or vcache.dtype != q.dtype:
        raise TypeError("q, kcache and vcache must be torch.float8_e4m3fn")
    if kscale.dtype != torch.float8_e4m3fn:
        raise TypeError("packed kscale must be the FP8 tail-row view")
    if q.dim() != 3:
        raise ValueError(f"q must be [total_q, Hq, D], got {tuple(q.shape)}")
    if kcache.dim() != 4 or vcache.dim() != 4:
        raise ValueError("kcache and vcache must be [num_pages, block_size, Hkv, D]")
    if kcache.shape != vcache.shape:
        raise ValueError(f"kcache/vcache shape mismatch: {kcache.shape} vs {vcache.shape}")
    if kscale.dim() != 4:
        raise ValueError("packed kscale must be [num_pages, scale_rows, Hkv, D]")

    total_q, num_head_q, head_dim = q.shape
    num_pages, block_size, num_head_kv, head_dim_kv = kcache.shape
    if head_dim != head_dim_kv:
        raise ValueError(f"q head_dim={head_dim} but cache head_dim={head_dim_kv}")
    if head_dim != config.head_dim:
        raise ValueError(f"current target is specialized for head_dim={config.head_dim}")
    if block_size != config.block_size:
        raise ValueError(f"current target is specialized for block_size={config.block_size}")
    if num_head_q % num_head_kv != 0:
        raise ValueError(f"Hq={num_head_q} must be divisible by Hkv={num_head_kv}")
    if (
        kscale.shape[0] != num_pages
        or kscale.shape[2] != num_head_kv
        or kscale.shape[3] != head_dim
    ):
        raise ValueError(f"packed kscale shape is incompatible: {tuple(kscale.shape)}")
    if kscale.shape[1] * head_dim != block_size * 4:
        raise ValueError("packed kscale tail rows must contain one FP32 word per KV token")
    if len({_storage_data_ptr(kcache), _storage_data_ptr(vcache), _storage_data_ptr(kscale)}) != 1:
        raise ValueError("kcache, vcache and packed kscale must be views into the same allocation")

    # Stride consistency (layout-agnostic — NHD / HND / cross-layer all OK):
    # K and V must share the same stride pattern, head_dim must be innermost
    # contiguous (TMA requirement), and the packed K-scale tail must reuse
    # K-cache's stride view.
    if kcache.stride() != vcache.stride():
        raise ValueError(
            f"kcache and vcache must share the same stride pattern (got "
            f"{kcache.stride()} vs {vcache.stride()}); mixing NHD/HND is "
            f"not supported in a single call."
        )
    if kcache.stride()[-1] != 1 or vcache.stride()[-1] != 1 or kscale.stride()[-1] != 1:
        raise ValueError(
            f"head_dim (innermost) must be contiguous: "
            f"kcache.stride={kcache.stride()}, vcache.stride={vcache.stride()}, "
            f"kscale.stride={kscale.stride()}"
        )
    if kscale.stride() != kcache.stride():
        raise ValueError(
            f"packed kscale tail must use the same physical layout as "
            f"kcache (got kscale.stride={kscale.stride()} vs "
            f"kcache.stride={kcache.stride()})"
        )

    if qscale.dtype != torch.float32 or vscale.dtype != torch.float32:
        raise TypeError("qscale and vscale must be float32")
    if (
        cu_seqlens_q.dtype != torch.int32
        or block_ids.dtype != torch.int32
        or seqlens_kvcache.dtype != torch.int32
    ):
        raise TypeError("cu_seqlens_q, block_ids and seqlens_kvcache must be int32")
    batch = cu_seqlens_q.numel() - 1
    if qscale.dim() != 3 or qscale.shape[:2] != (batch, num_head_q):
        raise ValueError("qscale must be [B, Hq, max_q]")
    if block_ids.shape[0] != batch or seqlens_kvcache.shape[0] != batch:
        raise ValueError("batch dimensions of block_ids/seqlens_kvcache/cu_seqlens_q do not match")
    if vscale.shape != (num_head_kv,):
        raise ValueError(f"vscale must be [Hkv], got {tuple(vscale.shape)}")
    if output is not None:
        if output.shape != (total_q, num_head_q, head_dim):
            raise ValueError(f"output shape mismatch: got {tuple(output.shape)}")
        _output_element_type(output)

    return total_q, num_head_q, num_head_kv, head_dim, block_size


_FORWARD_CACHE: dict[Fp8PagedPrefillConfig, Fp8PagedPrefillCuteDSL] = {}


def _get_forward(config: Fp8PagedPrefillConfig) -> Fp8PagedPrefillCuteDSL:
    runner = _FORWARD_CACHE.get(config)
    if runner is None:
        runner = Fp8PagedPrefillCuteDSL(config)
        _FORWARD_CACHE[config] = runner
    return runner


# Cache of the fixed p_scale_log2 tensor (filled with ``log2(256)``). Indexed
# by (device, num_head_q) so we don't allocate a tiny fp32 buffer on every call.
# The CutDSL FP8 prefill ABI intentionally does not expose runtime p_scale.
_DEFAULT_P_SCALE_LOG2_CACHE: dict = {}


def _default_p_scale_log2(num_head_q: int, device: torch.device) -> torch.Tensor:
    key = (str(device), int(num_head_q))
    cached = _DEFAULT_P_SCALE_LOG2_CACHE.get(key)
    if cached is None:
        cached = torch.full(
            (num_head_q,),
            float(DEFAULT_P_SCALE_LOG2),
            dtype=torch.float32,
            device=device,
        )
        _DEFAULT_P_SCALE_LOG2_CACHE[key] = cached
    return cached


def attention_prefill_fp8_cutedsl(
    q: torch.Tensor,
    kcache: torch.Tensor,
    vcache: torch.Tensor,
    qscale: torch.Tensor,
    kscale: torch.Tensor,
    vscale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_ids: torch.Tensor,
    seqlens_kvcache: torch.Tensor,
    max_seqlens_q: int,
    *,
    output: Optional[torch.Tensor] = None,
    config: Fp8PagedPrefillConfig = Fp8PagedPrefillConfig(),
) -> torch.Tensor:
    _validate_inputs(
        q,
        kcache,
        vcache,
        qscale,
        kscale,
        vscale,
        cu_seqlens_q,
        block_ids,
        seqlens_kvcache,
        output,
        config,
    )
    if int(max_seqlens_q) > qscale.shape[2]:
        raise ValueError(
            f"max_seqlens_q={max_seqlens_q} exceeds qscale padded extent {qscale.shape[2]}"
        )
    if int(max_seqlens_q) < config.tile_q or int(max_seqlens_q) % config.tile_q != 0:
        raise ValueError(
            "current CuteDSL target expects max_seqlens_q to be a positive "
            f"multiple of {config.tile_q}, got {max_seqlens_q}"
        )
    if output is None:
        output = torch.empty(
            (q.shape[0], q.shape[1], vcache.shape[-1]),
            dtype=torch.bfloat16,
            device=q.device,
        )

    num_head_q = q.shape[1]
    p_scale_log2 = _default_p_scale_log2(num_head_q, q.device)
    return _get_forward(config)(
        q,
        kcache,
        vcache,
        qscale,
        kscale,
        vscale,
        cu_seqlens_q,
        block_ids,
        seqlens_kvcache,
        int(max_seqlens_q),
        output,
        p_scale_log2,
    )
