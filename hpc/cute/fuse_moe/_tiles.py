# Copyright (C) 2026 Tencent.

"""Grouped-GEMM tile heights for the fused MoE, and how routing picks one."""

# Heights a grouped-GEMM tile can have, and so the granularity every expert's
# row block is padded to. Widest first, like the tile widths in
# ``_grouped_gemm``. Both are fractions of a scale-factor atom on purpose: with
# 256 experts even a prefill step routes only a few dozen rows to each of them,
# and a 128-row tile would spend most of the mainloop on padding. Rows 0..63 of
# an atom are a regular sub-layout of it, and so are rows 0..31 (see the mn
# split in ``hpc.cute._layouts``), so a shorter tile still addresses scales
# through the same descriptor. Buffers stay 128-row aligned either way. 16-row
# tiles are not addressable: the atom's MN mode splits on 32. 128-row tiles
# compile but the eight MMA warps all land on M and the A LDSM copy does not
# match.
GEMM_TILE_M_CHOICES = (64, 32)
# The tallest, which is what worst-case padding and buffer bounds go by.
GEMM_TILE_M = GEMM_TILE_M_CHOICES[0]
# Rows per expert from which the tall tile is used.
TALL_TILE_MIN_ROWS = 36


def gemm_tile_m_for(rows_per_expert: float) -> int:
    """Tile height whose padding costs less than the tiles it saves.

    Every expert's rows are rounded up to a whole tile, so a tile taller than
    what an expert actually got is padding the mainloop still reads; a tile
    shorter than that splits the expert's rows over more tiles, and every
    extra row tile streams that expert's weights again. Prefill is bound by
    that weight traffic, so the tall tile wins once an expert's rows fill a
    little over half of it. At 48 rows/expert on DeepSeek-V4 TP4 (2048
    tokens) one 64-row tile is 19% faster than two 32-row tiles; on Qwen TEP4
    the two are within 1% from 36 to 64 rows/expert. A 128-row tile is not
    used: the MMA warp split does not lower, and K=64 (six stages at
    tile_m=64) lost 2-3% to extra k-tiles.
    """
    short, tall = GEMM_TILE_M_CHOICES[-1], GEMM_TILE_M_CHOICES[0]
    return short if rows_per_expert < TALL_TILE_MIN_ROWS else tall
