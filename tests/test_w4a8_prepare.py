# Copyright (C) 2026 Tencent.

import pytest
import torch

from hpc.w4a8 import prepare_mxfp4_w4a8, prepare_w4a8_weights


def test_prepare_mxfp4_strict_scale_identity_and_views():
    packed = torch.arange(256, dtype=torch.uint8).repeat(2 * 64 * 64 // 256).reshape(2, 64, 64)
    scale = torch.full((2, 64, 4), 120, dtype=torch.uint8)
    scale[0, :, 1] = 131  # The full legal relative-exponent range, 1..12.
    scale[1, :, :] = 127
    weight, relative, common = prepare_mxfp4_w4a8(packed, scale)
    assert torch.equal(weight, packed)
    assert relative.dtype == torch.uint8 and common.dtype == torch.float32
    assert int(relative.min()) == 1 and int(relative.max()) == 12
    torch.testing.assert_close(common, torch.tensor([2.0**-2, 2.0**5]), rtol=0, atol=0)
    # q4 * 2**(rel-6) * common equals q4 * the original E8M0 group scale.
    restored = torch.exp2(relative.float() - 6) * common[:, None, None]
    torch.testing.assert_close(restored, torch.exp2(scale.float() - 127), rtol=0, atol=0)
    viewed = prepare_mxfp4_w4a8(packed.view(torch.int32), scale.view(torch.float8_e8m0fnu))
    for actual, expected in zip(viewed, (weight, relative, common)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_prepare_mxfp4_rejects_bad_exponents_and_requantization():
    packed = torch.zeros(2, 64, 64, dtype=torch.uint8)
    scales = torch.full((2, 64, 4), 120, dtype=torch.uint8)
    scales[0, 0, 0] = 132
    with pytest.raises(ValueError, match="exceeds 11"):
        prepare_mxfp4_w4a8(packed, scales)
    scales[0, 0, 0] = 255
    with pytest.raises(ValueError, match="255"):
        prepare_mxfp4_w4a8(packed, scales)
    with pytest.raises(NotImplementedError, match="strict"):
        prepare_mxfp4_w4a8(packed, scales, strict=False)


@pytest.mark.parametrize("n,k", [(64, 128), (128, 256)])
def test_prepare_w4a8_packed_operand_coordinates(n, k):
    generator = torch.Generator().manual_seed(431)
    raw = torch.randint(256, (2, n, k // 2), dtype=torch.uint8, generator=generator)
    # Force every FP4 code, including positive/negative zero, into the fixture.
    raw[:, 0, :8] = torch.tensor([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE])
    minimum = torch.tensor([0, 243], dtype=torch.int16)
    exponents = torch.arange(n * (k // 32)).reshape(1, n, k // 32) % 12
    exponents = (exponents + minimum[:, None, None]).to(torch.uint8)
    packed, relative, common = prepare_w4a8_weights(raw, exponents)
    assert packed.shape == (2, k // 64, n * 8) and packed.dtype == torch.int32
    assert relative.shape == (2, k // 32, n) and relative.dtype == torch.uint8
    assert common.shape == (2,) and common.dtype == torch.float32
    assert all(t.device == raw.device and t.is_contiguous() for t in (packed, relative, common))
    torch.testing.assert_close(common, torch.exp2(minimum.float() - 122), rtol=0, atol=0)

    # Independent coordinate oracle: CUTLASS GMMA::ALayout_64x32 uses
    # shape ((4,8,4),(4,2,2)), stride ((256,1,16),(64,8,1024)).
    # Inspect decoded FP4 bits, without using the production pack transforms.
    raw_values, words = raw.tolist(), packed.tolist()
    scales, source_scales = relative.reshape(2, -1).tolist(), exponents.tolist()
    for expert in range(2):
        for k64 in range(k // 64):
            for n64 in range(n // 64):
                for thread in range(128):
                    thread_offset = (thread % 4) * 256 + (thread // 4) % 8 + (thread // 32) * 16
                    for slot in range(4):
                        word = words[expert][k64][n64 * 512 + thread * 4 + slot] & 0xFFFFFFFF
                        for value in range(8):
                            fragment_index = value % 4 + (slot % 2) * 4 + (value // 4) * 8
                            offset = thread_offset + (fragment_index % 4) * 64
                            offset += ((fragment_index // 4) % 2) * 8 + (fragment_index // 8) * 1024
                            row = n64 * 64 + offset % 64
                            column = k64 * 64 + (slot // 2) * 32 + offset // 64
                            expected = (
                                raw_values[expert][row][column // 2] >> (column % 2 * 4)
                            ) & 15
                            magnitude = (word >> (value * 4)) & 7
                            sign_bit = value % 4 * 8 + (3 if value < 4 else 7)
                            actual = magnitude | (((word >> sign_bit) & 1) << 3)
                            assert actual == expected, (expert, row, column)
        # Decode each physical byte's coordinates independently of pack permute.
        for position, actual in enumerate(scales[expert]):
            rowpair = position % 2
            slab = position // 2 % 4
            lane_group = position // 8 % 8
            warp = position // 64 % 4
            n64 = position // 256 % (n // 64)
            k128 = position // (4 * n)
            row = n64 * 64 + warp * 16 + lane_group + rowpair * 8
            group = k128 * 4 + slab
            expected = source_scales[expert][row][group] - int(minimum[expert]) + 1
            assert actual == expected, (expert, row, group)
