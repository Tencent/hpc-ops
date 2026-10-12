# Copyright (C) 2026 Tencent.
#
# Adapted from the CUTLASS CuTe DSL example
# examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/
# dense_blockscaled_gemm_persistent_cooperative.py
# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""SM120 grouped MXFP8 x MXFP4 block-scaled GEMM.

The dense CUTLASS example is turned into a grouped GEMM by observing that all
four operands stay addressable from a single tensor descriptor once every
expert's row block is padded to a multiple of the tile height:

* ``A``/``C`` are one ``[total_padded_rows, K]`` / ``[total_padded_rows, N]``
  tensor. Expert ``e`` starts at row ``tile_m * cu_m_tiles[e]``, so a tile
  origin is always a whole number of tiles into the tensor.
* ``SFA`` is one buffer under ``tile_atom_to_shape_SF((total_padded_rows, K, 1))``.
  Tile alignment is exactly what makes a per-expert scale-factor base offset
  unnecessary: expert ``e``'s scales begin on a boundary that is a regular
  sub-layout of the atom, whichever tile height the launch picked.
* ``B``/``SFB`` are batched over experts, so the expert index is just the L
  coordinate of an ordinary 3-D tensor.

That removes the reason the grouped path would otherwise need per-group TMA
tensormap updates, so the mainloop, the MMA and the epilogue are the dense
example's, unchanged. Only the tile scheduler is new: it walks a flat work
index and binary-searches ``cu_m_tiles`` for the owning expert.

The cost is workspace: buffers are sized for the worst case
``total_routes + (tile_m - 1) * active_experts`` rows.

The kernel also has a variant that ends in SwiGLU and MXFP8 quantization
instead of a bfloat16 store, which is how the MoE's up projection is run: the
two halves SwiGLU multiplies are N/2 columns apart, so a CTA takes a tile of
each and the epilogue never has to look outside its own registers. See
``_build``'s ``fuse_swiglu``.

A second variant (``b_fp8`` + ``block128``) keeps the same mainloop and the
same MXFP8 MMA instruction, but B is E4M3 rather than packed E2M1 and the
real scales are Qwen's float32 128x128 blocks. Those cannot be rounded to
UE8M0, so the MMA's SFA/SFB fragments are filled with unit 1.0 in registers
and dummy scale-factor TMA is skipped. Each K=128 tile's accumulator is
multiplied by ``a_scale[row, k_tile] * w_scale[n_block, k_tile]`` before it
is added into the running sum. The fused SwiGLU epilogue uses a 128-wide
slab so one CTA owns a whole quantization block and can write float32
scales the way the standalone activation kernel did.

A third variant (``b_fp8`` without ``block128``) is the DeepGEMM SM120
recipe: B is still E4M3, but the 128-block float32 scales have been ceiled
to UE8M0 and broadcast onto the MX 32-vector layout, so the MMA applies
them in hardware. That reuses the V4 fused 64-wide mainloop -- three
pipeline stages instead of two -- which is the point of the conversion.
Prefill (an expert filling a 64-row tile) also shortens the K tile to 64
so A/B per stage halves and the fused epilogue still leaves room for a
deep pipeline; decode stays on K=128.
"""

from typing import Any, Dict, Optional, Tuple, Type

import torch

from .._compile import current_stream, import_cutlass, launcher, make_ptr, scalar
from .._layouts import (
    SCALE_K_ALIGNMENT,
    SCALE_MN_ALIGNMENT,
    SCALE_VECTOR_SIZE,
    ceil_div,
    scale_offset_device,
)
from .._quant import FP8_MAX, UNIT_UE8M0, build_helpers as build_quant_helpers
from ._tiles import GEMM_TILE_M

# A fraction of a scale-factor atom; ``_layouts`` explains which fractions are
# addressable and which one a launch gets, and ``make_sf_mma_layout`` below is
# what the choice costs. This is the tallest, and the one a caller that has no
# routing estimate to go on falls back to.
TILE_M = GEMM_TILE_M
TILE_K = 128
# Prefill FP8×FP8: half an SF atom along K, only as part of the prefill tile
# pair (see ``_prefill_tiles``).
TILE_K_PREFILL = 64

# Rows per expert from which FP8 x FP8 launches take the prefill tile pair,
# about 2048 tokens on Qwen TEP4. Measured against never switching: +1.3% at
# 2048 tokens and +2.4-3.6% from 2560 to 8192; a cutoff of 200 or 280 only
# gives up the 2048-3072 band.
PREFILL_FUSED_ROWS = 160.0

# Rows of A one TMA instruction brings in. A tile is loaded as ``tile_m //
# a_subtile`` of these rather than one box the height of the tile, so that a
# tile holding fewer routed rows than that can aim its leftover boxes back at
# the last one that has data. The byte count the pipeline's barrier waits for
# does not move, the repeated boxes were just read and hit L2, and DRAM only
# sees the rows an expert actually got. Decode is what this is for: 256 experts
# sharing 32 tokens leave nearly every tile with one live row out of 32, and
# the padding is then most of what the mainloop reads.
#
# The floor is 16, not 8: the sub-boxes all issue from the one DMA warp, and at
# eight of them per K tile the issue cost outweighs the traffic saved -- 6%
# slower than a whole-tile box at 2048 tokens, and slower even at 32. See
# ``_a_subtile_for`` for which height a launch gets.

# Tile widths the kernel is compiled for, widest first. A decode step gives
# every live expert exactly one row tile, so the tile count -- and with it how
# much of the GPU is busy -- is set by N alone: at 8 tokens the up-projection
# has 352 tiles for 110 SMs, and the ragged fourth wave costs more than the
# extra A traffic of the narrow tile. ``_tile_n_for`` picks between them.
TILE_N_CHOICES = (128, 64)
TILE_N = TILE_N_CHOICES[0]

# Slab width of the SwiGLU-fused kernel. Its tile is two of these, so it moves
# the same bytes per CTA as an unfused 128-wide tile and keeps the same three
# mainloop stages; a 128-wide slab would leave room for one.
FUSED_TILE_N = 64

# Qwen FP8 quantizes SwiGLU in 128-column blocks. Fusing that into GEMM1 needs
# the CTA to own a whole block, so the slab is an atom wide. Two 128-wide B
# slabs plus the fp32 epilogue tile leave two mainloop stages at tile_m=32 and
# one at tile_m=64; the 32-row tile is what decode and modest prefill use.
FUSED_TILE_N_FP8 = 128

# How much better the last wave has to fill before halving the tile is worth
# it. Splitting a tile leaves B traffic alone but reads A twice, which measures
# at a few percent, so a narrow tile that only ties on occupancy is a loss.
NARROW_TILE_MARGIN = 1.05

# The epilogue buffer is dead weight for all but the last k iteration of a tile,
# and it competes with the A/B pipeline for the same 99KB. Draining the tile in
# half-width slices costs one extra pair of barriers and buys a third mainloop
# stage, which is worth much more: at two stages the DMA warp cannot run more
# than one k tile ahead of the MMA warps.
EPI_TILE_N = 64
SF_VEC_SIZE = 32

# 16 bytes of BF16 per store instruction, the widest global store there is.
EPI_ELEMS_PER_THREAD = 8

# Widest BF16 atomic there is: red.global.add.noftz.bf16x2. Scalar BF16 has no
# atomic at all, and the f32 forms move twice the bytes for an accumulator whose
# rounding the output's own BF16 rounding hides anyway.
SCATTER_PAIR = 2

# Columns a thread scatters per pass over the epilogue tile, trading how well
# the atomics coalesce against how often the tile is walked. A warp holding 16
# bytes each spreads one atomic over four lines and covers a quarter of each; at
# 8 bytes it covers half of two. Down-projection K is short, so the epilogue is
# a large part of the kernel and the extra passes are not free: 8 columns
# measured 7% slower than 4, and 2 columns 1% slower.
SCATTER_ELEMS_PER_THREAD = 4


# One token index and one routing weight per row of a tile, so that the walk
# over the tile reads them from SMEM instead of chasing row -> route -> weight
# through global memory once per pass.
def scatter_table_bytes(tile_m: int) -> int:
    return tile_m * (32 // 8) * 2


# Columns one thread quantizes in the fused epilogue. Sixteen gives a 16-byte
# FP8 store and four threads to a 64-column tile row; with 256 threads that is
# one pass over a 64-row tile, and half the threads idle over a 32-row one.
ACT_ELEMS_PER_THREAD = 16

# Warps that run the MMA and the epilogue. A further warp drives the TMA loads.
MMA_WARPS = 8
WARP_SIZE = 32

# One CTA per SM, holding the whole SMEM. Marlin instead runs three thin CTAs
# per SM and keeps more of the memory system busy for it, but that needs a
# stage small enough to still afford four of them in a third of the SMEM, and
# ours cannot get there: the MMA reads FP4 out of eight-bit containers, so B
# alone costs twice what marlin's packed B does. Measured at two CTAs per SM
# with the 32-row tile, the fused GEMM1 is left with a single mainloop stage --
# 4% slower at 32 tokens and 18% at one.
CTAS_PER_SM = 1
# The thin form: half the MMA warps, two on M and two on N of a 64-row tile,
# and two CTAs per SM, so one CTA's epilogue runs under the other's mainloop.
THIN_TILE_M = 64
THIN_MMA_WARPS = 4
THIN_CTAS_PER_SM = 2
THIN_MMA_REGS = 200

# ldsm.b4x16_p64 puts the FP4 nibble in the low half of each byte while
# mma.sync.kind::mxf8f6f4 reads it from the middle, so the register operand is
# shifted left by two before the MMA. See cute::fp4_shift_B in
# cutlass/include/cute/atom/mma_traits_sm120.hpp.
#
# The shift runs on whole 32-bit registers rather than byte by byte: the unpack
# leaves the high nibble of every byte zero, so no bits cross a byte boundary
# and four bytes shift together. That is a quarter of the ALU work in the
# innermost loop, where it competes with the MMA for issue slots.
FP4_SHIFT_BITS = 2

# Gathered A (``a_gather``): tile rows come straight from a token-major
# activation buffer through TMA ``tile::gather4``, which loads one box row from
# each of four arbitrary tensor rows, instead of from a per-route copy of it.
# The box is one 128-byte row, a full line of the SW128 swizzle atom, so the
# four rows land at the offsets a plain TMA load of the tile would use.
A_GATHER_ROW_BYTES = 128
A_GATHER_ROWS = 4
# Each gather4 holds its issuing thread long enough that the DMA warp, issuing
# a 64-row tile's sixteen of them on top of the B and scale loads, falls behind
# the MMA: GEMM1 was a quarter slower at 8k tokens that way, and as slow with
# cp.async in place of gather4. Four warps that only gather keep it level up to
# 2k tokens; eight are no better than four. With the eight 232-register MMA
# warps and the DMA warp that is 64768 of the SM's 65536 registers, so there is
# no room for more.
A_GATHER_WARPS = 4

_cache: Dict[Tuple[Any, ...], Any] = {}
_stage_info: Dict[Tuple[Any, ...], int] = {}


def _build(
    num_sms: int,
    tile_n: int,
    fuse_swiglu: bool = False,
    scatter: bool = False,
    a_subtile: int = TILE_M,
    tile_m: int = TILE_M,
    b_fp8: bool = False,
    block128: bool = False,
    tile_k: int = TILE_K,
    a_gather: bool = False,
    thin: bool = False,
):
    cutlass, cute = import_cutlass()

    mma_warps = THIN_MMA_WARPS if thin else MMA_WARPS
    ctas_per_sm = THIN_CTAS_PER_SM if thin else CTAS_PER_SM
    a_subtiles_per_tile = tile_m // a_subtile
    if a_gather and (block128 or tile_k != A_GATHER_ROW_BYTES):
        raise ValueError(
            f"gathered A needs MX scales and {A_GATHER_ROW_BYTES}-byte K tiles; "
            f"got block128={block128}, tile_k={tile_k}"
        )
    # Thin with K=64 B tiles is an illegal instruction on MXFP4 B.
    if thin and (fuse_swiglu or a_gather or tile_m != THIN_TILE_M or tile_k != TILE_K):
        raise ValueError(
            f"the thin form is only the unfused GEMM2 on {THIN_TILE_M}x{TILE_K} tiles; "
            f"got fuse_swiglu={fuse_swiglu}, a_gather={a_gather}, "
            f"tile_m={tile_m}, tile_k={tile_k}"
        )

    import cuda.bindings.driver as cuda
    import cutlass.pipeline as pipeline
    import cutlass.utils as utils
    import cutlass.utils.blackwell_helpers as sm120_utils
    import cutlass.utils.blockscaled_layout as blockscaled_utils
    import cutlass.cute.math as cute_math
    import cutlass.utils.hopper_helpers as sm90_utils
    from cutlass.cute.nvgpu import cpasync

    swiglu, quantization_scale, _ = build_quant_helpers()

    def tma_gather4(atom, destination, barrier, column, rows):
        """Load box row ``column`` of four tensor rows into ``destination``.

        CuTe DSL has no copy op for ``tile::gather4`` yet, so this issues the
        PTX against the descriptor of a plain one-row TMA atom. A row of -1 is
        out of bounds: TMA fills it with zeros and still counts its bytes.
        """
        from cutlass._mlir.dialects import cute as cute_ir
        from cutlass._mlir.dialects import cute_nvgpu as cute_nvgpu_ir
        from cutlass._mlir.dialects import llvm
        from cutlass.cute.typing import AddressSpace

        exec_atom = cute_nvgpu_ir.atom_make_exec_tma(atom._trait.value)
        descriptor = cute_nvgpu_ir.get_tma_desc_addr(
            cute_ir.PtrType.get(
                cute_nvgpu_ir.TmaDescriptorTiledType.get(), AddressSpace.generic, 64
            ),
            exec_atom,
        )
        operands = [
            cutlass.Int32(destination.toint()).ir_value(),
            descriptor.toint().ir_value(),
            cutlass.Int32(barrier.toint()).ir_value(),
            cutlass.Int32(column).ir_value(),
        ] + [cutlass.Int32(row).ir_value() for row in rows]
        llvm.inline_asm(
            None,
            operands,
            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
            ".mbarrier::complete_tx::bytes [$0], [$1, {$3, $4, $5, $6, $7}], [$2];",
            "r,l,r,r,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    def mbarrier_complete_tx(barrier, byte_count):
        """Count ``byte_count`` bytes as arrived at ``barrier`` without moving any."""
        from cutlass._mlir.dialects import llvm

        llvm.inline_asm(
            None,
            [cutlass.Int32(barrier.toint()).ir_value(), cutlass.Int32(byte_count).ir_value()],
            "mbarrier.complete_tx.shared::cta.b64 [$0], $1;",
            "r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    # SwiGLU pairs column j of the gate half with column j + N/2 of the up half,
    # which no single tile of a plain N-major GEMM holds. Fusing it means giving
    # the CTA both halves: two B slabs sharing one A tile and one accumulator
    # each. Two 64-wide slabs cost exactly what one 128-wide tile costs, so the
    # mainloop is the unfused one -- only the epilogue is new.
    slabs = 2 if fuse_swiglu else 1
    if fuse_swiglu and block128 and tile_n != FUSED_TILE_N_FP8:
        raise ValueError(
            "fused 128-block FP8 SwiGLU needs a 128-wide slab so one CTA "
            "owns a whole quantization block"
        )

    def make_sf_mma_layout(tiled_mma, tile_mn, tile_k, sf_vec_size, num_stages):
        """The part of a staged scale atom that one tile's MMA reads.

        TMA always brings in the whole 128-element atom -- asking it for part of
        one turns the single descriptor into 32 eight-byte copies -- so a tile
        narrower than an atom takes its share in SMEM instead. An atom's mn mode
        splits as ``(32, 4):(16, 4)``, so any run of ``32 * j`` elements starting
        on a multiple of its own length is again a regular sub-layout, four bytes
        further in per 32 elements; that is all the caller has to shift by. Along
        K the atom is four scale groups of 32; a 64-wide A/B tile is a prefix
        of two. TMA still brings the whole atom; the MMA view shifts by
        ``tile_k // 32`` bytes for the second half. At ``tile_mn == 128`` and
        ``tile_k == 128`` this reproduces ``sm120_make_smem_layout_sfa/sfb``.
        """
        blk_mn, blk_sf = SCALE_MN_ALIGNMENT, 4
        assert tile_mn % 32 == 0 and blk_mn % tile_mn == 0, "tile must divide an atom by 32s"
        mma_nsf = tiled_mma.shape_mnk[2] // sf_vec_size
        atom_elems = blk_mn * blk_sf
        k_groups = tile_k // sf_vec_size
        assert tile_k % sf_vec_size == 0 and k_groups % mma_nsf == 0
        assert blk_sf % k_groups == 0, "tile_k must be a prefix of an SF atom along K"

        layout = cute.make_layout(
            (
                ((32, tile_mn // 32), 1),
                ((sf_vec_size, mma_nsf), k_groups // mma_nsf, 1),
            ),
            stride=(((16, 4), atom_elems), ((0, 1), mma_nsf, atom_elems)),
        )
        # Stage stride matches TMA's full-atom buffer, not the MMA prefix.
        return cute.append(layout, cute.make_layout(num_stages, stride=atom_elems))

    def group_k_modes(fragment, k_blocks):
        """Give the fragment's k index a mode of its own, at mode 2.

        ``partition_fragment_SF*`` lays a fragment out as value, N iterations
        and k blocks, but CuTe folds the last two together whenever their
        strides happen to nest -- which is exactly what splitting the warps
        four ways on N does to SFB. Rebuilding the layout from the halves of
        the folded mode undoes that; where nothing folded, the modes past the
        first two are grouped as before.
        """
        layout = fragment.layout
        if cute.rank(layout) > 2:
            return cute.group_modes(fragment, 2, cute.rank(fragment))
        value, folded = layout[0], layout[1]
        assert cute.rank(folded) == 2, "expected the k blocks folded into the N mode"
        assert cute.size(folded[1]) == k_blocks, "the folded tail is not the k blocks"
        return cute.make_tensor(
            fragment.iterator,
            cute.make_layout(
                (value.shape, folded[0].shape, folded[1].shape),
                stride=(value.stride, folded[0].stride, folded[1].stride),
            ),
        )

    @cute.jit
    def find_group(cu_m_tiles, num_groups, m_flat):
        """Last expert whose first tile index is at or before ``m_flat``."""
        low = cutlass.Int32(0)
        high = num_groups - 1
        while low < high:
            mid = (low + high + 1) // 2
            if cu_m_tiles[mid] <= m_flat:
                low = mid
            else:
                high = mid - 1
        return low

    @cute.jit
    def quantize_tile(
        sC,
        quant_frag,
        quant_out,
        row_in_tile,
        chunk,
        c_iter,
        c_row_stride,
        c_scale_ptr,
        k_atoms,
        m_flat,
        n_tile,
        valid_rows,
    ):
        """Quantize the SwiGLU tile sitting in SMEM, in whole rows.

        A thread owns ``ACT_ELEMS_PER_THREAD`` consecutive columns of one row,
        which is a fraction of a scale block; the threads holding the rest of
        the block are its immediate neighbours, so the block maximum is a
        butterfly over the low lane bits. MXFP8 writes a UE8M0 byte per 32
        columns; 128-block FP8 writes a float32 scale per 128 columns.
        Everything but the stores runs unpredicated because the shuffle needs
        the padding rows' threads too.
        """
        quant_cols = ACT_ELEMS_PER_THREAD
        # MXFP8 reduces 32-column UE8M0 blocks; Qwen FP8 reduces a 128-column
        # float32 block. Both are a power-of-two number of 16-element chunks, so
        # the block maximum is a butterfly over the low lane bits either way.
        scale_block = SCALE_K_ALIGNMENT if block128 else SCALE_VECTOR_SIZE
        chunks_per_block = scale_block // quant_cols
        butterfly_steps = chunks_per_block.bit_length() - 1
        rows_per_pass = mma_warps * WARP_SIZE // (tile_n // quant_cols)
        for pass_idx in cutlass.range_constexpr(max(1, tile_m // rows_per_pass)):
            row = row_in_tile + pass_idx * rows_per_pass
            # A tile shorter than one pass leaves the tail threads without a row.
            # They stay in the loop for the butterfly and reread the last row
            # instead, which keeps the read inside the tile.
            read_row = (
                row if cutlass.const_expr(tile_m >= rows_per_pass) else cutlass.min(row, tile_m - 1)
            )
            cute.autovec_copy(
                cute.make_tensor(
                    sC.iterator + (read_row * tile_n + chunk * quant_cols),
                    cute.make_layout(quant_cols),
                ),
                quant_frag,
            )
            block_max = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(quant_cols):
                block_max = cute.arch.fmax(block_max, cute_math.absf(quant_frag[i]))
            for step in cutlass.range_constexpr(butterfly_steps):
                block_max = cute.arch.fmax(
                    block_max, cute.arch.shuffle_sync_bfly(block_max, offset=1 << step)
                )
            if cutlass.const_expr(block128):
                scale = block_max / cutlass.Float32(FP8_MAX)
                inverse_scale = cutlass.Float32(1.0) / (scale + cutlass.Float32(1e-8))
                storage = cutlass.Int32(0)
            else:
                storage, inverse_scale = quantization_scale(block_max)
                scale = cutlass.Float32(0.0)

            if row < valid_rows:
                out_row = m_flat * tile_m + row
                column = n_tile * tile_n + chunk * quant_cols
                if chunk % chunks_per_block == 0:
                    if cutlass.const_expr(block128):
                        n_blocks = c_row_stride // SCALE_K_ALIGNMENT
                        slot = cute.make_tensor(
                            c_scale_ptr + cutlass.Int64(out_row) * n_blocks + n_tile,
                            cute.make_layout(1),
                        )
                        slot[0] = scale
                    else:
                        slot = cute.make_tensor(
                            c_scale_ptr + scale_offset_device(out_row, column, k_atoms),
                            cute.make_layout(1),
                        )
                        slot[0] = cutlass.Uint8(storage)
                for i in cutlass.range_constexpr(quant_cols):
                    quant_out[i] = cutlass.Float8E4M3FN(quant_frag[i] * inverse_scale)
                destination = cute.make_tensor(
                    (c_iter + cutlass.Int64(out_row) * c_row_stride + column).align(quant_cols),
                    cute.make_layout(quant_cols),
                )
                cute.autovec_copy(quant_out, destination)

    class Sm120GroupedBlockScaledGemm:
        """A = MXFP8 (or FP8), B = MXFP4 or FP8, optional float32 128-block scales."""

        def __init__(self):
            self.acc_dtype = cutlass.Float32
            self.sf_vec_size = SF_VEC_SIZE
            self.cluster_shape_mnk = (1, 1, 1)
            self.tile_shape_mnk = (tile_m, tile_n, tile_k)
            # The fused epilogue reduces along the row, so it needs the whole
            # tile in shared memory at once, and it needs it in fp32: the block
            # maximum is taken over the SwiGLU output, not over its rounding.
            self.epi_tile = (tile_m, tile_n if fuse_swiglu else min(EPI_TILE_N, tile_n))
            # The MMA atom is 16 rows, so the tile height is how many warps go on
            # M; the rest of the eight go on N. Thin, two warps go on each, and
            # each warp repeats the atom over a 32 x tile_n/2 share of the tile.
            if thin:
                self.atom_layout_mnk = (2, 2, 1)
            else:
                atom_m = tile_m // 16
                assert MMA_WARPS % atom_m == 0, "tile height must divide the MMA warps"
                self.atom_layout_mnk = (atom_m, MMA_WARPS // atom_m, 1)

            self.occupancy = ctas_per_sm
            self.num_mma_warps = mma_warps
            self.dma_warp_id = self.num_mma_warps
            self.num_threads_per_warp = WARP_SIZE
            load_warps = 1 + (A_GATHER_WARPS if a_gather else 0)
            self.threads_per_cta = (self.num_mma_warps + load_warps) * self.num_threads_per_warp
            self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")
            self.buffer_align_bytes = 1024

            self.mma_sync_barrier = pipeline.NamedBarrier(
                barrier_id=1,
                num_threads=self.num_mma_warps * self.num_threads_per_warp,
            )
            self.epilog_sync_barrier = pipeline.NamedBarrier(
                barrier_id=2,
                num_threads=self.num_mma_warps * self.num_threads_per_warp,
            )
            self.load_register_requirement = 40
            self.gather_register_requirement = 32
            self.mma_register_requirement = THIN_MMA_REGS if thin else 232

        def _setup_attributes(self):
            # MXFP4 B uses the mixed MMA and lives in Int8 containers so LDSM
            # can unpack nibbles. FP8 B is the same MMA kind as the dense WO
            # GEMM: both operands E4M3, SMEM and TMA see the values as stored.
            if b_fp8:
                mma_op = cute.nvgpu.warp.MmaMXF8Op(self.a_dtype, self.acc_dtype, self.sf_dtype)
                self.smem_alloc_a_dtype = self.a_dtype
                self.smem_alloc_b_dtype = self.b_dtype
                self.tma_internal_b_dtype = None
            else:
                mma_op = cute.nvgpu.warp.MmaMXF8F6F4Op(
                    self.a_dtype, self.b_dtype, self.acc_dtype, self.sf_dtype
                )
                self.smem_alloc_a_dtype = self.a_dtype
                self.smem_alloc_b_dtype = cutlass.Int8
                self.tma_internal_b_dtype = cutlass.Int8
            self.epi_dtype = cutlass.Float32 if fuse_swiglu else self.c_dtype

            permutation_mnk = sm120_utils.get_permutation_mnk(
                self.tile_shape_mnk, self.sf_vec_size, True
            )
            self.tiled_mma = cute.make_tiled_mma(
                mma_op, cute.make_layout(self.atom_layout_mnk), permutation_mnk=permutation_mnk
            )
            self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

            # Scales are staged, loaded and addressed by whole atoms; only the
            # MMA sees the half of one that matches the tile.
            self.sf_tile_shape_mnk = (
                SCALE_MN_ALIGNMENT,
                SCALE_MN_ALIGNMENT,
                SCALE_K_ALIGNMENT,
            )
            sfa_per_stage = blockscaled_utils.sm120_make_smem_layout_sfa(
                self.tiled_mma, self.sf_tile_shape_mnk, self.sf_vec_size, 1
            )
            sfb_per_stage = blockscaled_utils.sm120_make_smem_layout_sfb(
                self.tiled_mma, self.sf_tile_shape_mnk, self.sf_vec_size, 1
            )
            # The predicated store walks rows of a whole epilogue tile at once,
            # so the tile has to keep the CTA's full height.
            assert self.epi_tile[0] == self.tile_shape_mnk[0], "epilogue tile must be tile height"

            self.ab_stage, self.epi_stage = self._compute_stages(sfa_per_stage, sfb_per_stage)
            assert self.epi_stage > 0, "not enough shared memory for the epilogue"

            (
                self.a_smem_layout_staged,
                self.b_smem_layout_staged,
                self.sfa_smem_layout_staged,
                self.sfb_smem_layout_staged,
                self.epi_smem_layout_staged,
            ) = self._make_smem_layouts()

        def _compute_stages(self, sfa_smem_layout, sfb_smem_layout):
            """Deepest A/B pipeline that still fits, by exact byte accounting.

            The obvious budget -- capacity minus a round reserve for barriers --
            is off by about a kilobyte here, and a kilobyte is the difference
            between two stages and one. One stage means the DMA warp and the MMA
            warps take strict turns, so the mainloop runs at roughly half speed.
            Laying the SharedStorage members out exactly instead removes the
            guesswork; ``buffer_align_bytes`` padding is the whole discrepancy.
            """
            # One epilogue buffer: the barrier around the register-to-SMEM copy
            # is unconditional, so a second buffer would never be in flight
            # while the first drains, and the space goes to the mainloop.
            epi_stage = 1
            epi_bytes_per_stage = cute.size(self.epi_tile) * self.epi_dtype.width // 8

            a_shape = cute.slice_(self.tile_shape_mnk, (None, 0, None))
            b_shape = cute.slice_(self.tile_shape_mnk, (0, None, None))
            a_bytes_per_stage = cute.size(a_shape) * self.smem_alloc_a_dtype.width // 8
            b_bytes_per_stage = cute.size(b_shape) * self.smem_alloc_b_dtype.width // 8 * slabs
            # cosize, not size: a half-atom SFA tile leaves holes in its stage,
            # and the allocator has to reserve them.
            sfa_bytes_per_stage = (
                cute.cosize(cute.filter_zeros(sfa_smem_layout)) * self.sf_dtype.width // 8
            )
            sfb_bytes_per_stage = (
                cute.cosize(cute.filter_zeros(sfb_smem_layout)) * self.sf_dtype.width // 8 * slabs
            )

            def storage_bytes(ab_stage):
                align = self.buffer_align_bytes
                offset = ab_stage * 2 * (cutlass.Int64.width // 8)
                staged = (
                    a_bytes_per_stage,
                    b_bytes_per_stage,
                )
                if not block128:
                    staged += (sfa_bytes_per_stage, sfb_bytes_per_stage)
                for per_stage in staged:
                    offset = -(-offset // align) * align + per_stage * ab_stage
                if block128:
                    # Unit UE8M0 fragments are filled in registers. Keep one
                    # unstaged SF layout in SMEM so partition_fragment_SF* still
                    # has a descriptor, but do not replicate it per pipeline stage.
                    offset = -(-offset // align) * align + sfa_bytes_per_stage
                    offset = -(-offset // align) * align + sfb_bytes_per_stage
                return -(-offset // align) * align + epi_bytes_per_stage * epi_stage

            budget = self.smem_capacity // self.occupancy
            if scatter:
                budget -= scatter_table_bytes(tile_m)
            if block128:
                # Prefetched 128-block A scales, one float per tile row.
                budget -= tile_m * (cutlass.Float32.width // 8)
            ab_stage = 1
            while storage_bytes(ab_stage + 1) <= budget:
                ab_stage += 1
            _stage_info[
                (tile_m, tile_n, tile_k, fuse_swiglu, scatter, b_fp8, block128, a_gather, thin)
            ] = ab_stage
            return ab_stage, epi_stage

        def _make_smem_layouts(self):
            a_smem_shape = cute.slice_(self.tile_shape_mnk, (None, 0, None))
            a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
                sm90_utils.get_smem_layout_atom(
                    self.a_layout, self.smem_alloc_a_dtype, self.tile_shape_mnk[2]
                ),
                self.smem_alloc_a_dtype,
            )
            a_smem_layout_staged = cute.tile_to_shape(
                a_smem_layout_atom, cute.append(a_smem_shape, self.ab_stage), order=(0, 1, 2)
            )
            # What one sub-box of the load sees. Tiling M first makes every
            # a_subtile rows a whole number of layout atoms, so a sub-box has
            # the same shape and swizzle wherever in the tile it lands and the
            # TMA descriptor can be built from a one-box layout.
            self.a_sub_smem_layout_staged = cute.tile_to_shape(
                a_smem_layout_atom, (a_subtile, self.tile_shape_mnk[2], 1), order=(0, 1, 2)
            )

            b_smem_shape = cute.slice_(self.tile_shape_mnk, (0, None, None))
            b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
                sm90_utils.get_smem_layout_atom(
                    self.b_layout, self.smem_alloc_b_dtype, self.tile_shape_mnk[2]
                ),
                self.smem_alloc_b_dtype,
            )
            # Both B slabs live in the stage mode, two buffers per pipeline
            # stage, so the pipeline still counts stages the way it always did.
            b_smem_layout_staged = cute.tile_to_shape(
                b_smem_layout_atom,
                cute.append(b_smem_shape, self.ab_stage * slabs),
                order=(0, 1, 2),
            )

            sf_stages = 1 if block128 else self.ab_stage
            sfa_smem_layout_staged = blockscaled_utils.sm120_make_smem_layout_sfa(
                self.tiled_mma, self.sf_tile_shape_mnk, self.sf_vec_size, sf_stages
            )
            sfb_smem_layout_staged = blockscaled_utils.sm120_make_smem_layout_sfb(
                self.tiled_mma,
                self.sf_tile_shape_mnk,
                self.sf_vec_size,
                sf_stages * slabs,
            )
            self.sfa_mma_layout_staged = make_sf_mma_layout(
                self.tiled_mma,
                self.tile_shape_mnk[0],
                self.tile_shape_mnk[2],
                self.sf_vec_size,
                sf_stages,
            )
            self.sfb_mma_layout_staged = make_sf_mma_layout(
                self.tiled_mma,
                self.tile_shape_mnk[1],
                self.tile_shape_mnk[2],
                self.sf_vec_size,
                sf_stages * slabs,
            )

            if cutlass.const_expr(fuse_swiglu):
                # Row major and unswizzled: the tile is written through the MMA's
                # own C partitioning and read back in whole rows, and a swizzle
                # would only make the second of those harder. It still has to be
                # a composed layout, because that is what the kernel signature
                # says the epilogue layout is; the swizzle is the identity.
                epi_smem_layout_staged = cute.make_composed_layout(
                    cute.make_swizzle(0, 4, 3),
                    0,
                    cute.make_ordered_layout(
                        cute.append(self.epi_tile, self.epi_stage), order=(1, 0, 2)
                    ),
                )
            else:
                c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
                    sm90_utils.get_smem_layout_atom(self.c_layout, self.c_dtype, self.epi_tile[1]),
                    self.c_dtype,
                )
                epi_smem_layout_staged = cute.tile_to_shape(
                    c_smem_layout_atom,
                    cute.append(self.epi_tile, self.epi_stage),
                    order=(0, 1, 2),
                )
            return (
                a_smem_layout_staged,
                b_smem_layout_staged,
                sfa_smem_layout_staged,
                sfb_smem_layout_staged,
                epi_smem_layout_staged,
            )

        @staticmethod
        def _make_tma_atom(tensor, smem_layout_staged, smem_tile, internal_type=None):
            smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
            kwargs = dict(
                num_multicast=1,
            )
            if internal_type is not None:
                kwargs["internal_type"] = internal_type
            return cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                tensor,
                smem_layout,
                smem_tile,
                **kwargs,
            )

        @cute.jit
        def __call__(
            self,
            a_ptr: cute.Pointer,
            b_ptr: cute.Pointer,
            sfa_ptr: cute.Pointer,
            sfb_ptr: cute.Pointer,
            c_ptr: cute.Pointer,
            c_scale_ptr: cute.Pointer,
            cu_ptr: cute.Pointer,
            counts_ptr: cute.Pointer,
            route_ptr: cute.Pointer,
            topk_scale_ptr: cute.Pointer,
            m: cutlass.Int32,
            n: cutlass.Int32,
            k: cutlass.Int32,
            num_groups: cutlass.Int32,
            num_topk: cutlass.Int32,
            swiglu_limit: cutlass.Float32,
            a_scale128_ptr: cute.Pointer,
            b_scale128_ptr: cute.Pointer,
            k_scale_stride: cutlass.Int32,
            a_tokens: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            # Layouts are built here rather than on the host so that the problem
            # extents stay dynamic and one compiled kernel serves every shape.
            # Gathered, A holds one row per token rather than per padded row;
            # M still sizes SFA and the tile count, and the tile TMA descriptor
            # is built but never used.
            a = cute.make_tensor(a_ptr, cute.make_layout((m, k, 1), stride=(k, 1, 0)))
            b = cute.make_tensor(b_ptr, cute.make_layout((n, k, num_groups), stride=(k, 1, n * k)))
            # Fused, N counts both SwiGLU halves but the output holds only their
            # product, so C is half as wide as B and quantized.
            c_width = n // slabs
            # Scattering, C is the per-token output and has fewer rows than this
            # says; the tensor only ever supplies shapes, the epilogue addresses
            # it through the raw pointer.
            c = cute.make_tensor(c_ptr, cute.make_layout((m, c_width, 1), stride=(c_width, 1, 0)))
            # Both are indexed by route, of which there are at most as many as
            # there are padded rows.
            row_to_route = cute.make_tensor(route_ptr, cute.make_layout(m))
            topk_scale = cute.make_tensor(topk_scale_ptr, cute.make_layout(m))
            cu_m_tiles = cute.make_tensor(cu_ptr, cute.make_layout(num_groups + 1))
            row_counts = cute.make_tensor(counts_ptr, cute.make_layout(num_groups))
            n_tiles = c_width // tile_n
            k_blocks_128 = k // SCALE_K_ALIGNMENT
            n_blocks_128 = n // SCALE_K_ALIGNMENT
            a_scale128 = cute.make_tensor(
                a_scale128_ptr, cute.make_layout((m, k_blocks_128), stride=(k_blocks_128, 1))
            )
            b_scale128 = cute.make_tensor(
                b_scale128_ptr,
                cute.make_layout(
                    (num_groups, n_blocks_128, k_scale_stride),
                    stride=(n_blocks_128 * k_scale_stride, k_scale_stride, 1),
                ),
            )

            self.a_dtype = a.element_type
            self.b_dtype = b.element_type
            self.c_dtype = c.element_type
            self.sf_dtype = cutlass.Float8E8M0FNU

            self.a_layout = utils.LayoutEnum.from_tensor(a)
            self.b_layout = utils.LayoutEnum.from_tensor(b)
            self.c_layout = utils.LayoutEnum.from_tensor(c)

            self._setup_attributes()

            sfa_tensor = cute.make_tensor(
                sfa_ptr,
                blockscaled_utils.tile_atom_to_shape_SF(a.shape, self.sf_vec_size),
            )
            sfb_tensor = cute.make_tensor(
                sfb_ptr,
                blockscaled_utils.tile_atom_to_shape_SF(b.shape, self.sf_vec_size),
            )

            tma_atom_a, tma_tensor_a = self._make_tma_atom(
                a, self.a_sub_smem_layout_staged, (a_subtile, self.tile_shape_mnk[2])
            )
            tma_atom_a_rows = tma_atom_a
            if cutlass.const_expr(a_gather):
                tma_atom_a_rows, _ = cpasync.make_tiled_tma_atom(
                    cpasync.CopyBulkTensorTileG2SOp(),
                    cute.make_tensor(a_ptr, cute.make_layout((a_tokens, k), stride=(k, 1))),
                    cute.make_composed_layout(
                        self.a_smem_layout_staged.inner,
                        0,
                        cute.make_layout((1, tile_k), stride=(tile_k, 1)),
                    ),
                    (1, tile_k),
                )
            tma_atom_b, tma_tensor_b = self._make_tma_atom(
                b,
                self.b_smem_layout_staged,
                (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
                internal_type=self.tma_internal_b_dtype,
            )
            tma_atom_sfa, tma_tensor_sfa = self._make_tma_atom(
                sfa_tensor,
                self.sfa_smem_layout_staged,
                (self.sf_tile_shape_mnk[0], self.sf_tile_shape_mnk[2]),
                internal_type=cutlass.Int16,
            )
            tma_atom_sfb, tma_tensor_sfb = self._make_tma_atom(
                sfb_tensor,
                self.sfb_smem_layout_staged,
                (self.sf_tile_shape_mnk[1], self.sf_tile_shape_mnk[2]),
                internal_type=cutlass.Int16,
            )

            @cute.struct
            class SharedStorage:
                mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
                sA: cute.struct.Align[
                    cute.struct.MemRange[
                        self.smem_alloc_a_dtype, cute.cosize(self.a_smem_layout_staged)
                    ],
                    self.buffer_align_bytes,
                ]
                sB: cute.struct.Align[
                    cute.struct.MemRange[
                        self.smem_alloc_b_dtype, cute.cosize(self.b_smem_layout_staged)
                    ],
                    self.buffer_align_bytes,
                ]
                sSFA: cute.struct.Align[
                    cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                    self.buffer_align_bytes,
                ]
                sSFB: cute.struct.Align[
                    cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                    self.buffer_align_bytes,
                ]
                sC: cute.struct.Align[
                    cute.struct.MemRange[self.epi_dtype, cute.cosize(self.epi_smem_layout_staged)],
                    self.buffer_align_bytes,
                ]

            self.shared_storage = SharedStorage

            # No min_blocks_per_mp for the thin form's two CTAs: with it the
            # kernel hangs. The register request already fits two per SM.
            self.kernel(
                tma_atom_a,
                tma_tensor_a,
                tma_atom_b,
                tma_tensor_b,
                tma_atom_sfa,
                tma_tensor_sfa,
                tma_atom_sfb,
                tma_tensor_sfb,
                c,
                c_scale_ptr,
                swiglu_limit,
                cu_m_tiles,
                row_counts,
                row_to_route,
                topk_scale,
                num_groups,
                num_topk,
                n_tiles,
                a_scale128,
                b_scale128,
                tma_atom_a_rows,
                self.tiled_mma,
                self.cta_layout_mnk,
                self.a_smem_layout_staged,
                self.b_smem_layout_staged,
                self.sfa_smem_layout_staged,
                self.sfa_mma_layout_staged,
                self.sfb_smem_layout_staged,
                self.sfb_mma_layout_staged,
                self.epi_smem_layout_staged,
            ).launch(
                grid=[num_sms, 1, 1],
                block=[self.threads_per_cta, 1, 1],
                cluster=[1, 1, 1],
                stream=stream,
                use_pdl=True,
            )

        @cute.kernel
        def kernel(
            self,
            tma_atom_a: cute.CopyAtom,
            mA_mkl: cute.Tensor,
            tma_atom_b: cute.CopyAtom,
            mB_nkl: cute.Tensor,
            tma_atom_sfa: cute.CopyAtom,
            mSFA_mkl: cute.Tensor,
            tma_atom_sfb: cute.CopyAtom,
            mSFB_nkl: cute.Tensor,
            mC_mnl: cute.Tensor,
            c_scale_ptr: cute.Pointer,
            swiglu_limit: cutlass.Float32,
            cu_m_tiles: cute.Tensor,
            row_counts: cute.Tensor,
            row_to_route: cute.Tensor,
            topk_scale: cute.Tensor,
            num_groups: cutlass.Int32,
            num_topk: cutlass.Int32,
            n_tiles: cutlass.Int32,
            a_scale128: cute.Tensor,
            b_scale128: cute.Tensor,
            tma_atom_a_rows: cute.CopyAtom,
            tiled_mma: cute.TiledMma,
            cta_layout_mnk: cute.Layout,
            a_smem_layout_staged: cute.ComposedLayout,
            b_smem_layout_staged: cute.ComposedLayout,
            sfa_smem_layout_staged: cute.Layout,
            sfa_mma_layout_staged: cute.Layout,
            sfb_smem_layout_staged: cute.Layout,
            sfb_mma_layout_staged: cute.Layout,
            epi_smem_layout_staged: cute.ComposedLayout,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

            if warp_idx == 0:
                cpasync.prefetch_descriptor(tma_atom_a_rows if a_gather else tma_atom_a)
                cpasync.prefetch_descriptor(tma_atom_b)
                if cutlass.const_expr(not block128):
                    cpasync.prefetch_descriptor(tma_atom_sfa)
                    cpasync.prefetch_descriptor(tma_atom_sfb)

            a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
            b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
            sfa_smem_layout = cute.slice_(sfa_smem_layout_staged, (None, None, 0))
            sfb_smem_layout = cute.slice_(sfb_smem_layout_staged, (None, None, 0))
            # 128-block FP8 scales live in float32, not in the MMA's UE8M0
            # operands. Issuing dummy unit-scale TMA would still add those bytes
            # to the pipeline barrier and occupy the DMA warp.
            tma_copy_bytes = cute.size_in_bytes(
                self.a_dtype, a_smem_layout
            ) + slabs * cute.size_in_bytes(self.b_dtype, b_smem_layout)
            if cutlass.const_expr(not block128):
                tma_copy_bytes += cute.size_in_bytes(
                    self.sf_dtype, sfa_smem_layout
                ) + slabs * cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)

            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(self.shared_storage)
            mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

            if cutlass.const_expr(scatter):
                scatter_token = smem.allocate_tensor(
                    cutlass.Int32, cute.make_layout(tile_m), byte_alignment=16
                )
                scatter_weight = smem.allocate_tensor(
                    cutlass.Float32, cute.make_layout(tile_m), byte_alignment=16
                )
            if cutlass.const_expr(block128):
                sAScale = smem.allocate_tensor(
                    cutlass.Float32, cute.make_layout(tile_m), byte_alignment=16
                )

            mainloop_pipeline = pipeline.PipelineTmaAsync.create(
                num_stages=self.ab_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_warps),
                tx_count=tma_copy_bytes,
                barrier_storage=mainloop_pipeline_array_ptr,
                cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
            )

            sA = storage.sA.get_tensor(
                a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
            )
            sB = storage.sB.get_tensor(
                b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
            )
            sC = storage.sC.get_tensor(
                epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
            )
            sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
            sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
            # Which share of the staged atom a tile reads, and how far into the
            # atom that share starts: element tile_m sits at (tile_m // 32) * 4.
            sfa_tiles_per_atom = SCALE_MN_ALIGNMENT // self.tile_shape_mnk[0]
            sfa_half_offset = self.tile_shape_mnk[0] // 32 * 4
            sfb_tiles_per_atom = SCALE_MN_ALIGNMENT // self.tile_shape_mnk[1]
            sfb_half_offset = self.tile_shape_mnk[1] // 32 * 4
            # A 64-wide A/B tile is half an SF atom along K. TMA still loads
            # the whole atom; the MMA view shifts this many bytes for the
            # second half (consecutive K-groups have stride 1).
            sf_k_tiles_per_atom = SCALE_K_ALIGNMENT // tile_k
            sf_k_half_offset = tile_k // SF_VEC_SIZE
            sSFA_mma = cute.make_tensor(sSFA.iterator, sfa_mma_layout_staged)
            sSFB_mma = cute.make_tensor(sSFB.iterator, sfb_mma_layout_staged)

            gA_mkl = cute.local_tile(
                mA_mkl, (a_subtile, self.tile_shape_mnk[2]), (None, None, None)
            )
            gB_nkl = cute.local_tile(
                mB_nkl, cute.slice_(self.tile_shape_mnk, (0, None, None)), (None, None, None)
            )
            gSFA_mkl = cute.local_tile(
                mSFA_mkl, cute.slice_(self.sf_tile_shape_mnk, (None, 0, None)), (None, None, None)
            )
            gSFB_nkl = cute.local_tile(
                mSFB_nkl, cute.slice_(self.sf_tile_shape_mnk, (0, None, None)), (None, None, None)
            )
            gC_mnl = cute.local_tile(
                mC_mnl, cute.slice_(self.tile_shape_mnk, (None, None, 0)), (None, None, None)
            )

            thr_mma = tiled_mma.get_slice(tidx)

            cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
            sA_sub = cute.local_tile(sA, (a_subtile, self.tile_shape_mnk[2]), (None, None, None))
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                0,
                cta_layout,
                cute.group_modes(sA_sub, 0, 2),
                cute.group_modes(gA_mkl, 0, 2),
            )
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b,
                0,
                cta_layout,
                cute.group_modes(sB, 0, 2),
                cute.group_modes(gB_nkl, 0, 2),
            )
            tAsSFA, tAgSFA = cpasync.tma_partition(
                tma_atom_sfa,
                0,
                cta_layout,
                cute.group_modes(sSFA, 0, 2),
                cute.group_modes(gSFA_mkl, 0, 2),
            )
            tAsSFA = cute.filter_zeros(tAsSFA)
            tAgSFA = cute.filter_zeros(tAgSFA)
            tBsSFB, tBgSFB = cpasync.tma_partition(
                tma_atom_sfb,
                0,
                cta_layout,
                cute.group_modes(sSFB, 0, 2),
                cute.group_modes(gSFB_nkl, 0, 2),
            )
            tBsSFB = cute.filter_zeros(tBsSFB)
            tBgSFB = cute.filter_zeros(tBgSFB)

            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            k_blocks = self.tile_shape_mnk[2] // tiled_mma.shape_mnk[2]
            tCrSFA = sm120_utils.partition_fragment_SFA(sSFA_mma[None, None, 0], thr_mma, tidx)
            tCrSFA = group_k_modes(tCrSFA, k_blocks)

            tCgC = thr_mma.partition_C(gC_mnl)
            # One set per B slab: gate and up run through the same A tile and
            # the same instruction schedule, only their operands differ.
            tCrB = []
            tCrSFB = []
            accumulators = []
            for _ in cutlass.range_constexpr(slabs):
                tCrB.append(tiled_mma.make_fragment_B(tCsB[None, None, None, 0]))
                fragment_sfb = sm120_utils.partition_fragment_SFB(
                    sSFB_mma[None, None, 0], thr_mma, tidx
                )
                tCrSFB.append(group_k_modes(fragment_sfb, k_blocks))
                accumulators.append(cute.make_rmem_tensor(tCgC.shape[:3], self.acc_dtype))
            tCcC = thr_mma.partition_C(cute.make_identity_tensor((tile_m, tile_n)))
            running = []
            if cutlass.const_expr(block128):
                unit_sf = cutlass.Uint8(UNIT_UE8M0).bitcast(cutlass.Float8E8M0FNU)
                for i in cutlass.range_constexpr(cute.size(tCrSFA)):
                    tCrSFA[i] = unit_sf
                for slab in cutlass.range_constexpr(slabs):
                    running.append(cute.make_rmem_tensor(tCgC.shape[:3], self.acc_dtype))
                    for i in cutlass.range_constexpr(cute.size(tCrSFB[slab])):
                        tCrSFB[slab][i] = unit_sf

            cute.arch.sync_threads()

            k_tile_cnt = cute.size(gA_mkl, mode=[3])
            total_tiles = cu_m_tiles[num_groups] * n_tiles

            mainloop_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )
            mainloop_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )

            cute.arch.griddepcontrol_wait()

            if warp_idx < self.num_mma_warps:
                cute.arch.setmaxregister_increase(self.mma_register_requirement)
                num_k_blocks = cute.size(tCrA, mode=[2])

                atom_copy_ldmatrix_A = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                    self.a_dtype,
                )
                if cutlass.const_expr(b_fp8):
                    atom_copy_ldmatrix_B = cute.make_copy_atom(
                        cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                        self.b_dtype,
                    )
                else:
                    atom_copy_ldmatrix_B = cute.make_copy_atom(
                        cute.nvgpu.warp.LdMatrix8x16x8bOp(
                            transpose=False, num_matrices=4, unpack_bits=4
                        ),
                        cutlass.Int8,
                    )
                smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, tiled_mma)
                smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, tiled_mma)

                atom_copy_sf = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.sf_dtype)
                smem_tiled_copy_SFA = cute.make_tiled_copy(
                    atom_copy_sf,
                    sm120_utils.get_layoutSFA_TV(tiled_mma),
                    (
                        cute.size(tiled_mma.permutation_mnk[0]),
                        cute.size(tiled_mma.permutation_mnk[2]),
                    ),
                )
                smem_tiled_copy_SFB = cute.make_tiled_copy(
                    atom_copy_sf,
                    sm120_utils.get_layoutSFB_TV(tiled_mma),
                    (
                        cute.size(tiled_mma.permutation_mnk[1]),
                        cute.size(tiled_mma.permutation_mnk[2]),
                    ),
                )

                thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
                thr_copy_B = smem_tiled_copy_B.get_slice(tidx)
                tCsA_copy_view = thr_copy_A.partition_S(sA)
                tCrA_copy_view = thr_copy_A.retile(tCrA)
                tCsB_copy_view = thr_copy_B.partition_S(sB)
                tCrB_copy_view = [thr_copy_B.retile(fragment) for fragment in tCrB]

                thr_copy_SFA = smem_tiled_copy_SFA.get_slice(tidx)
                thr_copy_SFB = smem_tiled_copy_SFB.get_slice(tidx)
                tCrSFA_copy_view = thr_copy_SFA.retile(tCrSFA)
                tCrSFB_copy_view = [thr_copy_SFB.retile(fragment) for fragment in tCrSFB]

                epi_threads = self.num_mma_warps * self.num_threads_per_warp
                c_iter = mC_mnl.iterator
                c_row_stride = mC_mnl.shape[1]

                if cutlass.const_expr(fuse_swiglu):
                    # Writing the tile through the MMA's own C partitioning and
                    # reading it back in plain rows is what turns the epilogue's
                    # awkward thread mapping into somebody else's problem.
                    tCsC = thr_mma.partition_C(sC[None, None, 0])
                    quant_chunks = self.epi_tile[1] // ACT_ELEMS_PER_THREAD
                    quant_row_in_tile = cutlass.Int32(tidx) // quant_chunks
                    quant_chunk = cutlass.Int32(tidx) % quant_chunks
                    quant_frag = cute.make_rmem_tensor(
                        cute.make_layout(ACT_ELEMS_PER_THREAD), self.acc_dtype
                    )
                    quant_out = cute.make_rmem_tensor(
                        cute.make_layout(ACT_ELEMS_PER_THREAD), self.c_dtype
                    )
                    k_atoms = ceil_div(c_row_stride, SCALE_K_ALIGNMENT)
                else:
                    copy_atom_r2s = sm120_utils.sm120_get_smem_store_op(
                        self.c_layout, elem_ty_d=self.c_dtype, elem_ty_acc=self.acc_dtype
                    )
                    copy_atom_C = cute.make_copy_atom(
                        cute.nvgpu.warp.StMatrix8x8x16bOp(self.c_layout.is_m_major_c(), 2),
                        self.c_dtype,
                    )
                    tiled_copy_r2s = cute.make_tiled_copy_S(
                        copy_atom_r2s, cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
                    )
                    thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                    tRS_sD = thr_copy_r2s.partition_D(sC)
                    tRS_rAcc = tiled_copy_r2s.retile(accumulators[0])
                    tRS_rD_layout = cute.make_layout(cute.shape(thr_copy_r2s.partition_S(sC))[:3])
                    tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)

                    # Expert row blocks are padded up to whole tiles, and with 256
                    # experts most of a decode tile is padding. Storing the tile
                    # with one bulk TMA copy would write all of it; a plain tiled
                    # copy can drop the padding rows, which is most of C's traffic.
                    store_elems = SCATTER_ELEMS_PER_THREAD if scatter else EPI_ELEMS_PER_THREAD
                    store_cols = self.epi_tile[1] // store_elems
                    store_rows = epi_threads // store_cols
                    copy_atom_s2g = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        self.c_dtype,
                        num_bits_per_copy=store_elems * self.c_dtype.width,
                    )
                    tiled_copy_s2g = cute.make_tiled_copy_tv(
                        copy_atom_s2g,
                        cute.make_layout((store_rows, store_cols), stride=(store_cols, 1)),
                        cute.make_layout((1, store_elems)),
                    )
                    thr_copy_s2g = tiled_copy_s2g.get_slice(tidx)
                    tSG_sC = thr_copy_s2g.partition_S(sC)
                    store_frag = cute.make_rmem_tensor(cute.make_layout(store_elems), self.c_dtype)
                    if cutlass.const_expr(scatter):
                        # The atomic takes exactly one pair, so the columns a
                        # thread holds leave in that many pieces.
                        scatter_pair = cute.make_rmem_tensor(
                            cute.make_layout(SCATTER_PAIR), self.c_dtype
                        )
                    # The destination address is formed by hand rather than by
                    # partitioning C: the row stride of C is a runtime value, and
                    # through it CuTe stops proving the eight columns contiguous
                    # and falls back to storing them one BF16 at a time.
                    store_row_in_tile = cutlass.Int32(tidx) // store_cols
                    store_col_in_tile = (cutlass.Int32(tidx) % store_cols) * store_elems

                epi_buffer = cutlass.Int32(0)
                work = cutlass.Int32(bidx)
                while work < total_tiles:
                    m_flat = work // n_tiles
                    n_tile = work % n_tiles
                    group = find_group(cu_m_tiles, num_groups, m_flat)

                    for slab in cutlass.range_constexpr(slabs):
                        accumulators[slab].fill(0.0)
                    if cutlass.const_expr(block128):
                        for slab in cutlass.range_constexpr(slabs):
                            running[slab].fill(0.0)
                    mainloop_consumer_state.reset_count()

                    # TMA staged the whole atom; take the MN (and K) share this
                    # tile owns. A 64-wide A/B tile starts at k_half=0.
                    sfa_mn_off = (m_flat % sfa_tiles_per_atom) * sfa_half_offset
                    sfb_mn_off = [
                        ((n_tile + slab * n_tiles) % sfb_tiles_per_atom) * sfb_half_offset
                        for slab in range(slabs)
                    ]
                    tCsSFA_copy_view = thr_copy_SFA.partition_S(
                        cute.make_tensor(
                            sSFA.iterator + sfa_mn_off,
                            sfa_mma_layout_staged,
                        )
                    )
                    tCsSFB_copy_view = [
                        thr_copy_SFB.partition_S(
                            cute.make_tensor(
                                sSFB.iterator + sfb_mn_off[slab],
                                sfb_mma_layout_staged,
                            )
                        )
                        for slab in range(slabs)
                    ]

                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_state
                    )
                    mainloop_pipeline.consumer_wait(mainloop_consumer_state, peek_ab_full_status)
                    stage = mainloop_consumer_state.index
                    tCsA_p = tCsA_copy_view[None, None, None, stage]
                    tCsB_p = [
                        tCsB_copy_view[None, None, None, stage * slabs + slab]
                        for slab in range(slabs)
                    ]
                    if cutlass.const_expr(not block128):
                        tCsSFA_p = tCsSFA_copy_view[None, None, None, stage]
                        tCsSFB_p = [
                            tCsSFB_copy_view[slab][None, None, None, stage * slabs + slab]
                            for slab in range(slabs)
                        ]
                    cute.copy(
                        smem_tiled_copy_A, tCsA_p[None, None, 0], tCrA_copy_view[None, None, 0]
                    )
                    if cutlass.const_expr(not block128):
                        cute.copy(
                            smem_tiled_copy_SFA,
                            cute.filter_zeros(tCsSFA_p)[None, None, 0],
                            cute.filter_zeros(tCrSFA_copy_view)[None, None, 0],
                        )
                    for slab in cutlass.range_constexpr(slabs):
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[slab][None, None, 0],
                            tCrB_copy_view[slab][None, None, 0],
                        )
                        if cutlass.const_expr(not block128):
                            cute.copy(
                                smem_tiled_copy_SFB,
                                cute.filter_zeros(tCsSFB_p[slab])[None, None, 0],
                                cute.filter_zeros(tCrSFB_copy_view[slab])[None, None, 0],
                            )

                    if cutlass.const_expr(block128):
                        row = cutlass.Int32(tidx)
                        if row < tile_m:
                            sAScale[row] = a_scale128[m_flat * tile_m + row, 0]

                    for k_tile in cutlass.range(0, k_tile_cnt - 1, 1, unroll=1):
                        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                            k_block_next = 0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                            if k_block_idx == num_k_blocks - 1:
                                mainloop_pipeline.consumer_release(mainloop_consumer_state)
                                mainloop_consumer_state.advance()
                                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                                    mainloop_consumer_state
                                )
                                stage = mainloop_consumer_state.index
                                tCsA_p = tCsA_copy_view[None, None, None, stage]
                                tCsB_p = [
                                    tCsB_copy_view[None, None, None, stage * slabs + slab]
                                    for slab in range(slabs)
                                ]
                                if cutlass.const_expr(not block128):
                                    if cutlass.const_expr(tile_k < SCALE_K_ALIGNMENT):
                                        k_half = (
                                            (k_tile + 1) % sf_k_tiles_per_atom
                                        ) * sf_k_half_offset
                                        tCsSFA_copy_view = thr_copy_SFA.partition_S(
                                            cute.make_tensor(
                                                sSFA.iterator + sfa_mn_off + k_half,
                                                sfa_mma_layout_staged,
                                            )
                                        )
                                        tCsSFB_copy_view = [
                                            thr_copy_SFB.partition_S(
                                                cute.make_tensor(
                                                    sSFB.iterator + sfb_mn_off[slab] + k_half,
                                                    sfb_mma_layout_staged,
                                                )
                                            )
                                            for slab in range(slabs)
                                        ]
                                    tCsSFA_p = tCsSFA_copy_view[None, None, None, stage]
                                    tCsSFB_p = [
                                        tCsSFB_copy_view[slab][
                                            None, None, None, stage * slabs + slab
                                        ]
                                        for slab in range(slabs)
                                    ]
                                mainloop_pipeline.consumer_wait(
                                    mainloop_consumer_state, peek_ab_full_status
                                )

                            for slab in cutlass.range_constexpr(slabs):
                                if cutlass.const_expr(not b_fp8):
                                    b_view = cute.recast_tensor(
                                        tCrB[slab][None, None, k_block_idx], cutlass.Int32
                                    )
                                    for i in cutlass.range_constexpr(cute.size(b_view)):
                                        b_view[i] = b_view[i] << FP4_SHIFT_BITS
                                cute.gemm(
                                    tiled_mma,
                                    accumulators[slab],
                                    [
                                        tCrA[None, None, k_block_idx],
                                        tCrSFA[None, None, k_block_idx],
                                    ],
                                    [
                                        tCrB[slab][None, None, k_block_idx],
                                        tCrSFB[slab][None, None, k_block_idx],
                                    ],
                                    accumulators[slab],
                                )
                            cute.copy(
                                smem_tiled_copy_A,
                                tCsA_p[None, None, k_block_next],
                                tCrA_copy_view[None, None, k_block_next],
                            )
                            if cutlass.const_expr(not block128):
                                cute.copy(
                                    smem_tiled_copy_SFA,
                                    cute.filter_zeros(tCsSFA_p)[None, None, k_block_next],
                                    cute.filter_zeros(tCrSFA_copy_view)[None, None, k_block_next],
                                )
                            for slab in cutlass.range_constexpr(slabs):
                                cute.copy(
                                    smem_tiled_copy_B,
                                    tCsB_p[slab][None, None, k_block_next],
                                    tCrB_copy_view[slab][None, None, k_block_next],
                                )
                                if cutlass.const_expr(not block128):
                                    cute.copy(
                                        smem_tiled_copy_SFB,
                                        cute.filter_zeros(tCsSFB_p[slab])[None, None, k_block_next],
                                        cute.filter_zeros(tCrSFB_copy_view[slab])[
                                            None, None, k_block_next
                                        ],
                                    )

                        if cutlass.const_expr(block128):
                            self.mma_sync_barrier.arrive_and_wait()
                            for slab in cutlass.range_constexpr(slabs):
                                n_idx = n_tile + slab * n_tiles
                                n_block = n_idx * tile_n // SCALE_K_ALIGNMENT
                                b_s = b_scale128[group, n_block, k_tile]
                                for i in cutlass.range_constexpr(cute.size(accumulators[slab])):
                                    row = tCcC[i][0]
                                    running[slab][i] = (
                                        running[slab][i]
                                        + accumulators[slab][i] * sAScale[row] * b_s
                                    )
                                accumulators[slab].fill(0.0)
                            row = cutlass.Int32(tidx)
                            if row < tile_m:
                                sAScale[row] = a_scale128[m_flat * tile_m + row, k_tile + 1]

                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_next = 0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                        if k_block_idx == num_k_blocks - 1:
                            cute.arch.fence_proxy("async.shared", space="cta")
                            mainloop_pipeline.consumer_release(mainloop_consumer_state)
                            mainloop_consumer_state.advance()
                        if k_block_next > 0:
                            cute.copy(
                                smem_tiled_copy_A,
                                tCsA_p[None, None, k_block_next],
                                tCrA_copy_view[None, None, k_block_next],
                            )
                            if cutlass.const_expr(not block128):
                                cute.copy(
                                    smem_tiled_copy_SFA,
                                    cute.filter_zeros(tCsSFA_p)[None, None, k_block_next],
                                    cute.filter_zeros(tCrSFA_copy_view)[None, None, k_block_next],
                                )
                            for slab in cutlass.range_constexpr(slabs):
                                cute.copy(
                                    smem_tiled_copy_B,
                                    tCsB_p[slab][None, None, k_block_next],
                                    tCrB_copy_view[slab][None, None, k_block_next],
                                )
                                if cutlass.const_expr(not block128):
                                    cute.copy(
                                        smem_tiled_copy_SFB,
                                        cute.filter_zeros(tCsSFB_p[slab])[None, None, k_block_next],
                                        cute.filter_zeros(tCrSFB_copy_view[slab])[
                                            None, None, k_block_next
                                        ],
                                    )
                        for slab in cutlass.range_constexpr(slabs):
                            if cutlass.const_expr(not b_fp8):
                                b_view = cute.recast_tensor(
                                    tCrB[slab][None, None, k_block_idx], cutlass.Int32
                                )
                                for i in cutlass.range_constexpr(cute.size(b_view)):
                                    b_view[i] = b_view[i] << FP4_SHIFT_BITS
                            cute.gemm(
                                tiled_mma,
                                accumulators[slab],
                                [tCrA[None, None, k_block_idx], tCrSFA[None, None, k_block_idx]],
                                [
                                    tCrB[slab][None, None, k_block_idx],
                                    tCrSFB[slab][None, None, k_block_idx],
                                ],
                                accumulators[slab],
                            )

                    if cutlass.const_expr(block128):
                        last_k = k_tile_cnt - 1
                        self.mma_sync_barrier.arrive_and_wait()
                        for slab in cutlass.range_constexpr(slabs):
                            n_idx = n_tile + slab * n_tiles
                            n_block = n_idx * tile_n // SCALE_K_ALIGNMENT
                            b_s = b_scale128[group, n_block, last_k]
                            for i in cutlass.range_constexpr(cute.size(accumulators[slab])):
                                row = tCcC[i][0]
                                running[slab][i] = (
                                    running[slab][i] + accumulators[slab][i] * sAScale[row] * b_s
                                )
                            for i in cutlass.range_constexpr(cute.size(accumulators[slab])):
                                accumulators[slab][i] = running[slab][i]

                    # Rows of this tile holding real tokens rather than the
                    # padding that rounds the expert up to a whole tile. Capped
                    # at the tile: the epilogue can hand a thread a row past the
                    # tile's height, and an expert with more tiles to come would
                    # otherwise let it store over the next tile's rows.
                    valid_rows = cutlass.min(
                        row_counts[group] - (m_flat - cu_m_tiles[group]) * tile_m,
                        tile_m,
                    )

                    if cutlass.const_expr(fuse_swiglu):
                        # The tile leaves as fp32 through SMEM and comes back
                        # split by row, because the quantization block spans
                        # columns the MMA scattered across threads.
                        self.epilog_sync_barrier.arrive_and_wait()
                        for i in cutlass.range_constexpr(cute.size(accumulators[0])):
                            tCsC[i] = swiglu(accumulators[0][i], accumulators[1][i], swiglu_limit)
                        self.epilog_sync_barrier.arrive_and_wait()

                        quantize_tile(
                            sC,
                            quant_frag,
                            quant_out,
                            quant_row_in_tile,
                            quant_chunk,
                            c_iter,
                            c_row_stride,
                            c_scale_ptr,
                            k_atoms,
                            m_flat,
                            n_tile,
                            valid_rows,
                        )
                    else:
                        if cutlass.const_expr(scatter):
                            # Where each row of the tile is going, resolved once
                            # here rather than once per pass over the tile: the
                            # walk below would otherwise chase row -> route ->
                            # weight through global memory every time, and that
                            # dependent pair of loads costs more than the stores
                            # it feeds. The first barrier is for the passes of
                            # the previous tile still reading the table.
                            self.epilog_sync_barrier.arrive_and_wait()
                            # valid_rows counts to the end of the expert, not to
                            # the end of this tile, and the table only covers
                            # the tile.
                            table_rows = cutlass.Int32(
                                cutlass.select_(
                                    valid_rows < self.tile_shape_mnk[0],
                                    valid_rows,
                                    cutlass.Int32(self.tile_shape_mnk[0]),
                                )
                            )
                            row = cutlass.Int32(tidx)
                            while row < table_rows:
                                route = row_to_route[m_flat * self.tile_shape_mnk[0] + row]
                                scatter_token[row] = route // num_topk
                                scatter_weight[row] = topk_scale[route]
                                row += epi_threads
                            self.epilog_sync_barrier.arrive_and_wait()

                        tile_origin = (
                            cutlass.Int64(m_flat * self.tile_shape_mnk[0] + store_row_in_tile)
                            * c_row_stride
                            + n_tile * self.tile_shape_mnk[1]
                            + store_col_in_tile
                        )
                        epi_rest_m = self.tile_shape_mnk[0] // self.epi_tile[0]
                        epi_rest_n = self.tile_shape_mnk[1] // self.epi_tile[1]
                        mma_tile_m = self.tile_shape_mnk[0] // cute.size(tRS_rAcc, mode=[1])
                        mma_tile_n = self.tile_shape_mnk[1] // cute.size(tRS_rAcc, mode=[2])

                        for epi_m in cutlass.range_constexpr(epi_rest_m):
                            for epi_n in cutlass.range_constexpr(epi_rest_n):
                                mma_m_per_epi = self.epi_tile[0] // mma_tile_m
                                mma_n_per_epi = self.epi_tile[1] // mma_tile_n
                                for mma_n_in_epi in cutlass.range_constexpr(mma_n_per_epi):
                                    for mma_m_in_epi in cutlass.range_constexpr(mma_m_per_epi):
                                        mma_n = epi_n * mma_n_per_epi + mma_n_in_epi
                                        mma_m = epi_m * mma_m_per_epi + mma_m_in_epi
                                        dst = tRS_rD[(None, mma_m_in_epi, mma_n_in_epi)]
                                        src = tRS_rAcc[(None, mma_m, mma_n)]
                                        for elem in cutlass.range_constexpr(cute.size(dst)):
                                            dst[elem] = src[elem]

                                tRS_rD_out = cute.make_rmem_tensor(
                                    tRS_rD_layout.shape, self.c_dtype
                                )
                                tRS_rD_out.store(tRS_rD.load().to(self.c_dtype))

                                epi_buffer = (epi_buffer + 1) % cute.size(tRS_sD, mode=[3])
                                self.epilog_sync_barrier.arrive_and_wait()
                                cute.copy(
                                    tiled_copy_r2s,
                                    tRS_rD_out,
                                    tRS_sD[(None, None, None, epi_buffer)],
                                )
                                self.epilog_sync_barrier.arrive_and_wait()

                                for store_m in cutlass.range_constexpr(cute.size(tSG_sC, mode=[1])):
                                    row_in_tile = store_row_in_tile + store_m * store_rows
                                    if row_in_tile < valid_rows:
                                        cute.autovec_copy(
                                            tSG_sC[(None, store_m, 0, epi_buffer)],
                                            store_frag,
                                        )
                                        if cutlass.const_expr(scatter):
                                            # This row is one expert's share of
                                            # one token, so weight it and add it
                                            # into that token's output here
                                            # rather than park it for a second
                                            # pass to sum.
                                            weight = scatter_weight[row_in_tile]
                                            base = (
                                                cutlass.Int64(scatter_token[row_in_tile])
                                                * c_row_stride
                                                + n_tile * self.tile_shape_mnk[1]
                                                + epi_n * self.epi_tile[1]
                                                + store_col_in_tile
                                            )
                                            for pair in cutlass.range_constexpr(
                                                store_elems // SCATTER_PAIR
                                            ):
                                                for i in cutlass.range_constexpr(SCATTER_PAIR):
                                                    scatter_pair[i] = self.c_dtype(
                                                        cutlass.Float32(
                                                            store_frag[pair * SCATTER_PAIR + i]
                                                        )
                                                        * weight
                                                    )
                                                cute.arch.atomic_add(
                                                    c_iter + (base + pair * SCATTER_PAIR),
                                                    scatter_pair.load().ir_value(),
                                                    scope="gpu",
                                                )
                                        else:
                                            # N is a multiple of the tile width
                                            # and each thread owns a whole
                                            # 16-byte column chunk, so the
                                            # address is aligned even though the
                                            # runtime row stride hides that from
                                            # CuTe.
                                            destination = cute.make_tensor(
                                                (
                                                    c_iter
                                                    + (
                                                        tile_origin
                                                        + epi_n * self.epi_tile[1]
                                                        + cutlass.Int64(store_m * store_rows)
                                                        * c_row_stride
                                                    )
                                                ).align(EPI_ELEMS_PER_THREAD * 2),
                                                cute.make_layout(EPI_ELEMS_PER_THREAD),
                                            )
                                            cute.autovec_copy(store_frag, destination)

                    work += num_sms

            elif warp_idx == self.dma_warp_id:
                cute.arch.setmaxregister_decrease(self.load_register_requirement)

                work = cutlass.Int32(bidx)
                while work < total_tiles:
                    m_flat = work // n_tiles
                    n_tile = work % n_tiles
                    group = find_group(cu_m_tiles, num_groups, m_flat)

                    # Sub-boxes past the expert's last routed row repeat that
                    # row's box: same bytes for the barrier, no new lines from
                    # DRAM. See ``a_subtile``.
                    valid_rows = row_counts[group] - (m_flat - cu_m_tiles[group]) * tile_m
                    last_sub = cutlass.min(
                        (valid_rows - 1) // a_subtile,
                        cutlass.Int32(a_subtiles_per_tile - 1),
                    )
                    sub_base = m_flat * a_subtiles_per_tile
                    tAgSFA_k = tAgSFA[(None, m_flat // sfa_tiles_per_atom, None, 0)]
                    # Fused, the second slab is the same tile of the up half of
                    # B, one whole gate half further along N.
                    tBgB_k = [
                        tBgB[(None, n_tile + slab * n_tiles, None, group)] for slab in range(slabs)
                    ]
                    tBgSFB_k = [
                        tBgSFB[(None, (n_tile + slab * n_tiles) // sfb_tiles_per_atom, None, group)]
                        for slab in range(slabs)
                    ]

                    mainloop_producer_state.reset_count()
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        mainloop_pipeline.producer_acquire(mainloop_producer_state)
                        barrier = mainloop_pipeline.producer_get_barrier(mainloop_producer_state)
                        count = mainloop_producer_state.count
                        stage = mainloop_producer_state.index
                        if cutlass.const_expr(not a_gather):
                            for sub in cutlass.range_constexpr(a_subtiles_per_tile):
                                cute.copy(
                                    tma_atom_a,
                                    tAgA[
                                        (
                                            None,
                                            sub_base + cutlass.min(cutlass.Int32(sub), last_sub),
                                            count,
                                            0,
                                        )
                                    ],
                                    tAsA[(None, sub, 0, stage)],
                                    tma_bar_ptr=barrier,
                                )
                        if cutlass.const_expr(not block128):
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA_k[(None, count * tile_k // SCALE_K_ALIGNMENT)],
                                tAsSFA[(None, stage)],
                                tma_bar_ptr=barrier,
                            )
                        for slab in cutlass.range_constexpr(slabs):
                            cute.copy(
                                tma_atom_b,
                                tBgB_k[slab][(None, count)],
                                tBsB[(None, stage * slabs + slab)],
                                tma_bar_ptr=barrier,
                            )
                            if cutlass.const_expr(not block128):
                                cute.copy(
                                    tma_atom_sfb,
                                    tBgSFB_k[slab][(None, count * tile_k // SCALE_K_ALIGNMENT)],
                                    tBsSFB[(None, stage * slabs + slab)],
                                    tma_bar_ptr=barrier,
                                )
                        mainloop_pipeline.producer_commit(mainloop_producer_state)
                        mainloop_producer_state.advance()

                    work += num_sms

                mainloop_pipeline.producer_tail(mainloop_producer_state)

            elif warp_idx > self.dma_warp_id:
                if cutlass.const_expr(a_gather):
                    cute.arch.setmaxregister_decrease(self.gather_register_requirement)
                    # The DMA warp's expect_tx already counts these bytes, and
                    # the phase cannot complete before that arrival however
                    # early they land, so the gather warps never arrive.
                    rows_per_warp = tile_m // A_GATHER_WARPS
                    lane = cutlass.Int32(tidx) % WARP_SIZE
                    row0 = (warp_idx - self.dma_warp_id - 1) * rows_per_warp + lane * A_GATHER_ROWS
                    issuing = lane < rows_per_warp // A_GATHER_ROWS
                    tokens = cute.make_rmem_tensor(cute.make_layout(A_GATHER_ROWS), cutlass.Int32)
                    sA_base = storage.sA.data_ptr()
                    gather_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer, self.ab_stage
                    )

                    work = cutlass.Int32(bidx)
                    while work < total_tiles:
                        m_flat = work // n_tiles
                        group = find_group(cu_m_tiles, num_groups, m_flat)
                        valid_rows = cutlass.min(
                            row_counts[group] - (m_flat - cu_m_tiles[group]) * tile_m,
                            cutlass.Int32(tile_m),
                        )
                        # Rows past the expert's last routed one are zero-filled
                        # where they share a gather with routed rows. Gathers
                        # with none are not issued, their bytes only counted, and
                        # the rows keep whatever the stage held: like the padding
                        # of a copied A they reach only rows the epilogue skips.
                        # Issuing them anyway made GEMM1 44-95% slower at two to
                        # eight tokens, where nearly every gather is one.
                        live = cutlass.Boolean(False)
                        if issuing:
                            live = row0 < valid_rows
                        for i in cutlass.range_constexpr(A_GATHER_ROWS):
                            tokens[i] = cutlass.Int32(-1)
                            if live:
                                if row0 + i < valid_rows:
                                    tokens[i] = row_to_route[m_flat * tile_m + row0 + i] // num_topk

                        gather_state.reset_count()
                        for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                            mainloop_pipeline.sync_object_empty.wait(
                                gather_state.index, gather_state.phase
                            )
                            barrier = mainloop_pipeline.producer_get_barrier(gather_state)
                            if live:
                                tma_gather4(
                                    tma_atom_a_rows,
                                    sA_base
                                    + (gather_state.index * tile_m + row0) * A_GATHER_ROW_BYTES,
                                    barrier,
                                    gather_state.count * tile_k,
                                    [tokens[i] for i in range(A_GATHER_ROWS)],
                                )
                            elif issuing:
                                mbarrier_complete_tx(barrier, A_GATHER_ROWS * A_GATHER_ROW_BYTES)
                            gather_state.advance()

                        work += num_sms

            cute.arch.griddepcontrol_launch_dependents()

    return Sm120GroupedBlockScaledGemm()


def _compiled(
    num_sms: int,
    tile_n: int,
    fuse_swiglu: bool,
    scatter: bool,
    a_subtile: int,
    tile_m: int,
    b_fp8: bool,
    block128: bool,
    tile_k: int,
    a_gather: bool,
    *args,
    thin: bool = False,
):
    key = (
        "grouped_gemm",
        num_sms,
        tile_n,
        fuse_swiglu,
        scatter,
        a_subtile,
        tile_m,
        b_fp8,
        block128,
        tile_k,
        a_gather,
        thin,
    )
    if key not in _cache:
        _, cute = import_cutlass()
        _cache[key] = launcher(
            cute.compile(
                _build(
                    num_sms,
                    tile_n,
                    fuse_swiglu,
                    scatter,
                    a_subtile,
                    tile_m,
                    b_fp8,
                    block128,
                    tile_k,
                    a_gather,
                    thin,
                ).__call__,
                *args,
            )
        )
    return _cache[key]


def _prefill_tiles(rows_per_expert: float, b_fp8: bool, block128: bool) -> bool:
    """Whether an FP8 x FP8 MX launch takes the prefill tile pair.

    The pair is a 128-wide fused GEMM1 slab with K=64 on both GEMMs. Either
    half alone loses (K=64 by 2-3%, the wide slab by 11-15%); together the
    extra stages pay for the extra k-tiles once experts hold enough rows. MXFP4
    B and the exact 128-block path were not measured this way and stay put.
    """
    return b_fp8 and not block128 and rows_per_expert >= PREFILL_FUSED_ROWS


def _tile_k_for(rows_per_expert: float, k: int, b_fp8: bool, block128: bool) -> int:
    """K tile: 128, or half an SF atom for the prefill tile pair."""
    return TILE_K_PREFILL if _prefill_tiles(rows_per_expert, b_fp8, block128) else TILE_K


def _fused_tile_n_for(rows_per_expert: float, b_fp8: bool, block128: bool) -> int:
    """Slab width of the SwiGLU-fused GEMM1.

    Fused 128-block FP8 needs the slab to be a whole quantization block. MX
    keeps a 64-wide slab so two of them cost an unfused 128-wide tile, except
    in the prefill tile pair.
    """
    if block128 or _prefill_tiles(rows_per_expert, b_fp8, block128):
        return FUSED_TILE_N_FP8
    return FUSED_TILE_N


def _a_subtile_for(rows_per_expert: float, tile_m: int) -> int:
    """Load box height that pays for itself at this much padding.

    An expert's rows are padded up to the tile height, so the shorter its row
    block the more of a whole-tile A load is padding, and the more a narrower
    box saves. What it costs is DMA-warp issue slots, which is why the box stops
    tracking the row count once the tile is mostly full. Measured against a
    whole-tile box on this model's shapes: 16 rows is 5% faster at a couple of
    rows per expert, 32 rows 2-3% faster from there to a few dozen, and past
    that neither is better than loading the tile in one go.
    """
    if rows_per_expert >= tile_m // 2:
        return tile_m
    # Sub-boxes below 16 rows lose to DMA-warp issue cost (measured 6% slower
    # than a whole-tile load even at 32 tokens).
    if rows_per_expert >= 2.0:
        short = min(32, tile_m // 2)
        if short >= 16:
            return short
    return min(tile_m, 16)


def _tile_n_for(m_tiles: int, n: int, num_sms: int) -> int:
    """Tile width whose wave tail wastes the least of the GPU.

    The kernel is persistent, so a launch costs ``ceil(tiles / num_sms)`` waves
    and every SM idle in the last one is lost time. That is what the narrow
    tile buys back, and only where the tail is ragged: at one token the wide
    tile leaves more than half the SMs with nothing to do, while past a few
    dozen tokens both widths fill their last wave and the narrow one is just
    extra traffic.
    """

    def occupancy(tile_n):
        tiles = m_tiles * (n // tile_n)
        return tiles / (-(-tiles // num_sms) * num_sms)

    best = TILE_N_CHOICES[0]
    target = occupancy(best) * NARROW_TILE_MARGIN
    for tile_n in TILE_N_CHOICES[1:]:
        if occupancy(tile_n) > target:
            best = tile_n
    return best


def _use_thin(fused: bool, tile_m: int, b_fp8: bool = False) -> bool:
    """Whether the GEMM2 launch takes the thin form: two CTAs of four MMA warps.

    Only the unfused GEMM2 with MXFP4 B on the 64-row tile. There one CTA's
    epilogue, scatter or plain, runs under the other's mainloop: 4-7% faster
    at 4-8k tokens on DeepSeek-V4 TP4, neutral below 2k. With FP8 B the
    halved SMEM leaves a single mainloop stage, which costs more than the
    overlap wins back (5-6% slower on Qwen at the same sizes).
    """
    return (not fused) and (not b_fp8) and tile_m == THIN_TILE_M


def grouped_block_scaled_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    c: torch.Tensor,
    cu_m_tiles: torch.Tensor,
    row_counts: torch.Tensor,
    num_groups: int,
    expected_m_tiles: Optional[int] = None,
    expected_rows_per_expert: Optional[float] = None,
    tile_m: int = TILE_M,
    c_scale: Optional[torch.Tensor] = None,
    swiglu_limit: float = -1.0,
    row_to_route: Optional[torch.Tensor] = None,
    topk_scale: Optional[torch.Tensor] = None,
    num_topk: int = 0,
    a_scale128: Optional[torch.Tensor] = None,
    b_scale128: Optional[torch.Tensor] = None,
    a_row_to_route: Optional[torch.Tensor] = None,
) -> None:
    """Run ``C = (A * SFA) @ (B * SFB)^T`` grouped over experts.

    Args:
        a: ``[total_padded_rows, K]`` E4M3 activations, or ``[num_tokens, K]``
            when ``a_row_to_route`` is given.
        b: ``[num_groups, N, K // 2]`` packed E2M1 weights, ``uint8``; or
            ``[num_groups, N, K]`` E4M3 when ``a_scale128`` is given.
        sfa: Flat UE8M0 buffer laid out as
            ``tile_atom_to_shape_SF((total_padded_rows, K, 1))``.
        sfb: Flat UE8M0 buffer laid out as
            ``tile_atom_to_shape_SF((N, K, num_groups))``.
        c: ``[total_padded_rows, N]`` BF16 output. Rows past each expert's real
            row count are left untouched.
        cu_m_tiles: ``int32[num_groups + 1]``, cumulative ``tile_m``-row tile counts.
        row_counts: ``int32[num_groups]``, real (unpadded) rows per expert.
        num_groups: Number of experts.
        expected_m_tiles: How many row tiles the launch is likely to have, used
            only to pick the tile width. It has to be a host-side guess: the
            real count lives in ``cu_m_tiles`` on the device, and reading it
            would serialise the pipeline and break CUDA graph capture. Defaults
            to the buffer height, which is a large overestimate and so always
            picks the widest tile.
        expected_rows_per_expert: Host-side guess at how many rows an expert
            gets, for the same reason, used only to pick the A load box height.
            Defaults to a full tile, which loads it in one go as before.
        tile_m: Rows in a tile. Not a free choice here: it is the granularity
            the routing kernel padded each expert's rows to, so it has to be
            the ``tile_m`` that ``RouteBuffers`` settled on.
        c_scale: Fuses SwiGLU and quantization into the epilogue when given.
            ``c`` is then ``[total_padded_rows, N // 2]`` E4M3 holding
            ``silu(gate) * up``. For MXFP8, ``c_scale`` is UE8M0 laid out as
            ``tile_atom_to_shape_SF((total_padded_rows, N // 2, 1))``. For
            128-block FP8 it is ``float32[total_padded_rows, (N // 2) // 128]``.
            The two SwiGLU halves have to meet inside a CTA, so the tile covers
            ``tile_n`` columns of each and the launch has half as many tiles.
        swiglu_limit: Clamp applied before the activation, or negative for none.
        row_to_route: Optional ``int32[total_padded_rows]`` mapping each row back
            to the route that put it there. Given, the epilogue weights every
            row by ``topk_scale`` and adds it into ``c`` at ``route //
            num_topk``, so ``c`` is ``[num_tokens, N]`` and holds the finished
            top-k sum; the caller must have initialized it. Rows arrive from
            different experts in no order, so the accumulation is atomic and the
            result is not bit-reproducible between runs.
        topk_scale: ``float32[num_tokens * num_topk]`` routing weights.
        num_topk: Routes per token.
        a_scale128: Optional ``float32[total_padded_rows, K // 128]`` Qwen
            activation scales. Given, B must be E4M3 and each K=128 tile is
            multiplied by ``a_scale128[row, k] * b_scale128[e, n//128, k]``
            after the MMA; SFA/SFB should be unit UE8M0 and are not TMA'd.
            Combined with ``c_scale``, GEMM1 fuses SwiGLU and 128-column FP8
            requantization the way the MXFP8 path fuses 32-column UE8M0.
        b_scale128: ``float32[num_groups, N // 128, K_stride]`` weight scales.
            ``K_stride`` may be padded to a multiple of 4.
        a_row_to_route: Optional ``int32[total_padded_rows]`` route of every
            padded row. Given, ``a`` holds one row per token and the kernel
            reads row ``r`` of the tile from ``a[a_row_to_route[r] //
            num_topk]`` instead of from a per-route copy; ``sfa`` is still per
            padded row. Needs MX scales and K tiles of 128.
    """
    cutlass, _ = import_cutlass()

    a_gather = a_row_to_route is not None
    k = a.shape[1]
    total_rows = a_row_to_route.shape[0] if a_gather else a.shape[0]
    if a_gather and num_topk <= 0:
        raise ValueError("a_row_to_route needs num_topk")
    if a_gather and not (a.is_contiguous() and a.data_ptr() % 16 == 0):
        raise ValueError("gathered A must be contiguous and 16-byte aligned")
    fused = c_scale is not None
    scatter = row_to_route is not None
    if a_gather and scatter:
        raise ValueError("gathered A and the scatter epilogue are not combined")
    if scatter and (topk_scale is None or num_topk <= 0):
        raise ValueError("row_to_route needs topk_scale and num_topk")
    if scatter and fused:
        raise ValueError("the fused SwiGLU epilogue has no scatter form")
    n = c.shape[1] * 2 if fused else c.shape[1]
    block128 = a_scale128 is not None
    b_fp8 = b.dtype == torch.float8_e4m3fn
    if block128 and not b_fp8:
        raise ValueError("128-block scales require FP8 B")
    if block128 and b_scale128 is None:
        raise ValueError("a_scale128 needs b_scale128")
    # FP8 B with real MX SFA/SFB is the DeepGEMM-style path: 128-block float32
    # scales have already been packed as UE8M0. The exact path passes
    # a_scale128 instead.
    # The tile is half an atom but the scale-factor layout is built from whole
    # ones, so the buffer -- not the tile -- sets the alignment.
    if total_rows % SCALE_MN_ALIGNMENT != 0:
        raise ValueError(f"A rows must be a multiple of {SCALE_MN_ALIGNMENT}; got {total_rows}")
    if n % SCALE_MN_ALIGNMENT != 0:
        raise ValueError(f"N must be a multiple of {SCALE_MN_ALIGNMENT}; got {n}")
    if k % SCALE_K_ALIGNMENT != 0:
        raise ValueError(f"K must be a multiple of {SCALE_K_ALIGNMENT}; got {k}")
    expected_b = (num_groups, n, k) if b_fp8 else (num_groups, n, k // 2)
    if tuple(b.shape) != expected_b:
        raise ValueError(f"B must be {expected_b}; got {tuple(b.shape)}")
    k_blocks = k // SCALE_K_ALIGNMENT
    n_blocks = n // SCALE_K_ALIGNMENT
    k_scale_stride = 1
    if block128:
        if tuple(a_scale128.shape) != (total_rows, k_blocks):
            raise ValueError(
                f"a_scale128 must be [{total_rows}, {k_blocks}]; got {tuple(a_scale128.shape)}"
            )
        if a_scale128.dtype != torch.float32 or b_scale128.dtype != torch.float32:
            raise ValueError("128-block scales must be float32")
        if b_scale128.shape[0] != num_groups or b_scale128.shape[1] != n_blocks:
            raise ValueError(
                f"b_scale128 must be [{num_groups}, {n_blocks}, K_stride]; "
                f"got {tuple(b_scale128.shape)}"
            )
        k_scale_stride = int(b_scale128.shape[2])
        if k_scale_stride < k_blocks:
            raise ValueError(
                f"b_scale128 K stride {k_scale_stride} is shorter than {k_blocks} K blocks"
            )
        if fused:
            out_blocks = (n // 2) // SCALE_K_ALIGNMENT
            if tuple(c_scale.shape) != (total_rows, out_blocks):
                raise ValueError(
                    f"c_scale must be [{total_rows}, {out_blocks}]; got {tuple(c_scale.shape)}"
                )
            if c_scale.dtype != torch.float32:
                raise ValueError("fused 128-block c_scale must be float32")

    thin = _use_thin(fused, tile_m, b_fp8)
    # Both the grid and the persistent loop's stride, so this is also how many
    # CTAs land on an SM.
    num_sms = _sm_count(a.device) * (THIN_CTAS_PER_SM if thin else CTAS_PER_SM)
    if expected_m_tiles is None:
        expected_m_tiles = total_rows // tile_m
    if expected_rows_per_expert is None:
        expected_rows_per_expert = float(tile_m)
    if fused:
        tile_n = _fused_tile_n_for(expected_rows_per_expert, b_fp8, block128)
    else:
        tile_n = _tile_n_for(expected_m_tiles, n, num_sms)
    a_subtile = _a_subtile_for(expected_rows_per_expert, tile_m)
    tile_k = _tile_k_for(expected_rows_per_expert, k, b_fp8, block128)

    args = (
        make_ptr(cutlass.Float8E4M3FN, a),
        make_ptr(cutlass.Float8E4M3FN if b_fp8 else cutlass.Float4E2M1FN, b),
        make_ptr(cutlass.Float8E8M0FNU, sfa),
        make_ptr(cutlass.Float8E8M0FNU, sfb),
        make_ptr(cutlass.Float8E4M3FN if fused else cutlass.BFloat16, c),
        make_ptr(
            cutlass.Float32 if fused and block128 else cutlass.Uint8,
            c_scale if fused else cu_m_tiles,
        ),
        make_ptr(cutlass.Int32, cu_m_tiles),
        make_ptr(cutlass.Int32, row_counts),
        make_ptr(
            cutlass.Int32,
            row_to_route if scatter else a_row_to_route if a_gather else cu_m_tiles,
        ),
        make_ptr(cutlass.Float32, topk_scale if scatter else cu_m_tiles),
        scalar(cutlass.Int32, total_rows),
        scalar(cutlass.Int32, n),
        scalar(cutlass.Int32, k),
        scalar(cutlass.Int32, num_groups),
        scalar(cutlass.Int32, num_topk),
        scalar(cutlass.Float32, swiglu_limit),
        make_ptr(cutlass.Float32, a_scale128 if block128 else cu_m_tiles),
        make_ptr(cutlass.Float32, b_scale128 if block128 else cu_m_tiles),
        scalar(cutlass.Int32, k_scale_stride),
        scalar(cutlass.Int32, a.shape[0]),
        current_stream(),
    )
    _compiled(
        num_sms,
        tile_n,
        fused,
        scatter,
        a_subtile,
        tile_m,
        b_fp8,
        block128,
        tile_k,
        a_gather,
        *args,
        thin=thin,
    )(*args)


_sm_counts: Dict[Any, int] = {}


def _sm_count(device) -> int:
    """SM count of ``device``, cached: querying properties costs microseconds."""
    count = _sm_counts.get(device)
    if count is None:
        count = torch.cuda.get_device_properties(device).multi_processor_count
        _sm_counts[device] = count
    return count
