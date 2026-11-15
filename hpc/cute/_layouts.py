# Copyright (C) 2026 Tencent.

"""Interleaved UE8M0 scale-factor layout for SM120 block-scaled GEMM.

Packed weight scales and activation SFA/SFB buffers must follow this layout.
The atom matches CUTLASS' ``Sm1xxBlockScaledConfig<32>``:

.. code-block:: cpp

    using SfKMajorAtom = Layout<Shape < Shape<_32,_4>, Shape<_32,_4>>,
                                Stride<Stride<_16,_4>, Stride< _0,_1>>>;
    tile_to_shape(SfAtom{}, make_shape(MN, K, L), Step<_2,_1,_3>{})

One atom covers a 128 x 128 (MN x K) block and occupies 512 bytes. Within an
atom the MN coordinate splits as ``mn = mn0 + 32 * mn1`` contributing
``mn0 * 16 + mn1 * 4``, and the K coordinate splits as ``k = k0 + 32 * k1``
contributing ``k1`` -- the inner 32 elements share one scale, which is exactly
why their stride is zero. ``Step<_2,_1,_3>`` makes the K block index the
fastest-varying rest mode.
"""

from typing import Any, Tuple

SCALE_VECTOR_SIZE = 32
SCALE_MN_ALIGNMENT = 128
SCALE_K_ALIGNMENT = 128

# Bytes covered by one 128x128 scale-factor atom.
SCALE_ATOM_BYTES = SCALE_MN_ALIGNMENT * SCALE_K_ALIGNMENT // SCALE_VECTOR_SIZE


def round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def scale_storage_size(rows: int, cols: int) -> int:
    """Bytes of the interleaved UE8M0 buffer for a logical ``[rows, cols]`` matrix.

    Pads ``rows`` and ``cols`` to 128 and divides by 32 (one byte per scale).
    """
    return (
        round_up(rows, SCALE_MN_ALIGNMENT) * round_up(cols, SCALE_K_ALIGNMENT) // SCALE_VECTOR_SIZE
    )


def scale_offset(mn: int, k_element: int, k_extent: int) -> int:
    """Byte offset of the scale covering ``[mn, k_element]``.

    Args:
        mn: Row index for SFA, column index for SFB.
        k_element: Reduction-dimension element index. Must be a multiple of
            :data:`SCALE_VECTOR_SIZE`; the inner 32 elements share one scale.
        k_extent: Logical reduction extent of the matrix, which fixes how many
            atoms one MN block spans.

    The MN extent deliberately does not participate: it only bounds the valid
    range, it does not change the offset of a valid coordinate.
    """
    mn_in_atom, mn_block = mn % SCALE_MN_ALIGNMENT, mn // SCALE_MN_ALIGNMENT
    k_in_atom, k_block = (
        k_element % SCALE_K_ALIGNMENT,
        k_element // SCALE_K_ALIGNMENT,
    )
    within_atom = (mn_in_atom % 32) * 16 + (mn_in_atom // 32) * 4 + k_in_atom // SCALE_VECTOR_SIZE
    k_atoms = ceil_div(k_extent, SCALE_K_ALIGNMENT)
    return within_atom + k_block * SCALE_ATOM_BYTES + mn_block * SCALE_ATOM_BYTES * k_atoms


def make_sf_layout(shape: Tuple[Any, ...]) -> Any:
    """Build the DSL scale-factor layout for an operand of ``shape``.

    ``shape`` is ``(M, K, L)`` for SFA and ``(N, K, L)`` for SFB. Must be called
    from inside a ``@cute.jit`` or ``@cute.kernel`` trace.
    """
    from cutlass.utils.blockscaled_layout import tile_atom_to_shape_SF

    return tile_atom_to_shape_SF(shape, SCALE_VECTOR_SIZE)


def scale_offset_device(mn: Any, k_element: Any, k_atoms: Any) -> Any:
    """Device-side counterpart of :func:`scale_offset`.

    ``k_atoms`` is ``ceil_div(k_extent, 128)``, hoisted by the caller because it
    is loop-invariant. Kept arithmetic rather than layout-based so the
    elementwise kernels do not have to materialize a layout per route.
    """
    mn_in_atom = mn % SCALE_MN_ALIGNMENT
    mn_block = mn // SCALE_MN_ALIGNMENT
    k_in_atom = k_element % SCALE_K_ALIGNMENT
    k_block = k_element // SCALE_K_ALIGNMENT
    within_atom = (mn_in_atom % 32) * 16 + (mn_in_atom // 32) * 4 + k_in_atom // SCALE_VECTOR_SIZE
    return within_atom + k_block * SCALE_ATOM_BYTES + mn_block * SCALE_ATOM_BYTES * k_atoms
