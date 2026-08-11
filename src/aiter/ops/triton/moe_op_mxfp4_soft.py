# SPDX-License-Identifier: MIT

import functools
import os
from pathlib import Path

import torch
import triton
import triton.language as tl


_MXFP4_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

_SIGNED_INT16_0X81C0 = tl.constexpr(0x81C0 - 0x10000)


@functools.lru_cache(maxsize=1)
def _has_compatible_triton_compiler() -> bool:
    explicit = os.getenv("TRITON_HIP_CLANG_PATH")
    if explicit:
        return Path(explicit).is_file()
    version = tuple(int(part) for part in triton.__version__.split(".")[:2])
    if version <= (3, 1):
        return Path("/opt/dtk/llvm/bin/clang").is_file()
    return Path("/opt/dtk/aillvm/bin/clang").is_file()


def _backend() -> str:
    backend = os.getenv("MXFP4_GFX936_BACKEND", "safe")
    if backend not in ("safe", "triton"):
        raise ValueError(f"unknown gfx936 MXFP4 backend: {backend}")
    return backend


def _require_triton_compiler() -> None:
    if not _has_compatible_triton_compiler():
        raise RuntimeError(
            "MXFP4_GFX936_BACKEND=triton requires a compatible compiler; "
            "set TRITON_HIP_CLANG_PATH explicitly"
        )


@triton.jit
def _fused_moe_mxfp4_bf16_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    b_mx_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_bmxe,
    stride_bmxn,
    stride_bmxk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return

    token_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    routed_tokens = tl.load(sorted_token_ids_ptr + token_offsets)
    token_mask = routed_tokens < num_valid_tokens
    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = c_ptr + routed_tokens[:, None] * stride_cm + offs_n[None, :] * stride_cn
    output_mask = token_mask[:, None] & (offs_n[None, :] < N)
    if expert == -1:
        tl.store(output_ptrs, 0.0, mask=output_mask)
        return

    pair_offsets = tl.arange(0, BLOCK_K // 2)
    a_ptrs = (
        a_ptr
        + (routed_tokens[:, None] // top_k) * stride_am
        + (2 * pair_offsets[None, :]) * stride_ak
    )
    b_ptrs = (
        b_ptr
        + expert * stride_be
        + pair_offsets[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )
    b_mx_scale_ptrs = (
        b_mx_scale_ptr
        + expert * stride_bmxe
        + (pair_offsets[:, None] // 16) * stride_bmxk
        + offs_n[None, :] * stride_bmxn
    )

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    fp4_to_bf16_scale = 0x7E800000
    fp4_to_bf16_scale = fp4_to_bf16_scale.to(tl.float32, bitcast=True).to(tl.bfloat16)
    for _ in range(K // BLOCK_K):
        packed = tl.load(
            b_ptrs,
            mask=offs_n[None, :] < N,
            other=0,
        ).to(tl.int8, bitcast=True).to(tl.int16)
        scale_codes = tl.load(
            b_mx_scale_ptrs,
            mask=offs_n[None, :] < N,
            other=127,
        ).to(tl.int16)
        scale_bits = tl.where(scale_codes == 0, 0x0040, scale_codes << 7)
        scale_bits = tl.where(scale_codes == 255, 0x7FC0, scale_bits).to(tl.int16)
        block_scales = scale_bits.to(tl.bfloat16, bitcast=True)

        low_bits = (((packed << 12).to(tl.int16) >> 6) & _SIGNED_INT16_0X81C0).to(tl.int16)
        high_bits = ((packed << 2).to(tl.int16) & _SIGNED_INT16_0X81C0).to(tl.int16)
        low = low_bits.to(tl.bfloat16, bitcast=True) * fp4_to_bf16_scale * block_scales
        high = high_bits.to(tl.bfloat16, bitcast=True) * fp4_to_bf16_scale * block_scales

        a_even = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0).to(tl.bfloat16)
        a_odd = tl.load(a_ptrs + stride_ak, mask=token_mask[:, None], other=0.0).to(tl.bfloat16)
        a_tile = tl.interleave(a_even, a_odd)
        weight_tile = tl.trans(tl.interleave(tl.trans(low), tl.trans(high)))
        accumulator += tl.dot(a_tile, weight_tile)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk
        b_mx_scale_ptrs += (BLOCK_K // 32) * stride_bmxk

    accumulator *= tl.load(a_scale_ptr) * tl.load(b_scale_ptr + expert)
    if MUL_ROUTED_WEIGHT:
        route_weights = tl.load(topk_weights_ptr + routed_tokens, mask=token_mask, other=0.0)
        accumulator *= route_weights[:, None]
    tl.store(output_ptrs, accumulator, mask=output_mask)


@triton.jit
def _linear_mxfp4_bf16_kernel(
    x_ptr,
    weight_ptr,
    scale_ptr,
    output_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_sn,
    stride_sk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    pair_offsets = tl.arange(0, BLOCK_K // 2)
    x_ptrs = (
        x_ptr
        + offs_m[:, None] * stride_xm
        + (2 * pair_offsets[None, :]) * stride_xk
    )
    weight_ptrs = (
        weight_ptr
        + pair_offsets[:, None] * stride_wk
        + offs_n[None, :] * stride_wn
    )
    scale_ptrs = (
        scale_ptr
        + (pair_offsets[:, None] // 16) * stride_sk
        + offs_n[None, :] * stride_sn
    )
    mask_m = offs_m < M
    mask_n = offs_n < N

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    fp4_to_bf16_scale = 0x7E800000
    fp4_to_bf16_scale = fp4_to_bf16_scale.to(tl.float32, bitcast=True).to(
        tl.bfloat16
    )
    for _ in range(K // BLOCK_K):
        packed = tl.load(
            weight_ptrs,
            mask=mask_n[None, :],
            other=0,
        ).to(tl.int8, bitcast=True).to(tl.int16)
        scale_codes = tl.load(
            scale_ptrs,
            mask=mask_n[None, :],
            other=127,
        ).to(tl.int16)
        scale_bits = tl.where(scale_codes == 0, 0x0040, scale_codes << 7)
        scale_bits = tl.where(scale_codes == 255, 0x7FC0, scale_bits).to(tl.int16)
        block_scales = scale_bits.to(tl.bfloat16, bitcast=True)

        low_bits = (
            ((packed << 12).to(tl.int16) >> 6) & _SIGNED_INT16_0X81C0
        ).to(tl.int16)
        high_bits = ((packed << 2).to(tl.int16) & _SIGNED_INT16_0X81C0).to(
            tl.int16
        )
        low = (
            low_bits.to(tl.bfloat16, bitcast=True)
            * fp4_to_bf16_scale
            * block_scales
        )
        high = (
            high_bits.to(tl.bfloat16, bitcast=True)
            * fp4_to_bf16_scale
            * block_scales
        )
        x_even = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0).to(tl.bfloat16)
        x_odd = tl.load(
            x_ptrs + stride_xk, mask=mask_m[:, None], other=0.0
        ).to(tl.bfloat16)
        x_tile = tl.interleave(x_even, x_odd)
        weight_tile = tl.trans(tl.interleave(tl.trans(low), tl.trans(high)))
        accumulator += tl.dot(x_tile, weight_tile)

        x_ptrs += BLOCK_K * stride_xk
        weight_ptrs += (BLOCK_K // 2) * stride_wk
        scale_ptrs += (BLOCK_K // 32) * stride_sk

    output_ptrs = (
        output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    )
    tl.store(output_ptrs, accumulator, mask=mask_m[:, None] & mask_n[None, :])


def dequant_mxfp4(
    packed: torch.Tensor,
    block_scales: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize packed OCP MXFP4 E2M1 values with E8M0 block scales."""
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed must be uint8, got {packed.dtype}")
    if block_scales.dtype != torch.uint8:
        raise TypeError(f"block_scales must be uint8, got {block_scales.dtype}")

    logical_k = packed.shape[-1] * 2
    if logical_k % 32 != 0:
        raise ValueError(f"logical K must be divisible by 32, got {logical_k}")
    if block_scales.shape[:-1] != packed.shape[:-1]:
        raise ValueError("packed and block_scales leading dimensions must match")
    if block_scales.shape[-1] * 32 != logical_k:
        raise ValueError("one E8M0 scale is required for every 32 MXFP4 values")
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)
    table = torch.tensor(_MXFP4_E2M1_VALUES, device=packed.device, dtype=torch.float32)
    values = table[codes.long()]

    scale_codes = block_scales.repeat_interleave(32, dim=-1)
    exponents = scale_codes.to(torch.int16) - 127
    scales = torch.exp2(exponents.to(torch.float32))
    scales = torch.where(
        scale_codes == 255,
        torch.full_like(scales, float("nan")),
        scales,
    )
    return (values * scales).to(dtype)


def linear_mxfp4_soft(
    x: torch.Tensor,
    packed: torch.Tensor,
    block_scales: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weight-only OCP MXFP4 linear fallback with an explicit backend."""
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"x must be fp16 or bf16, got {x.dtype}")
    if packed.dtype != torch.uint8 or block_scales.dtype != torch.uint8:
        raise TypeError("packed weights and E8M0 scales must be uint8")
    if packed.ndim != 2 or block_scales.ndim != 2:
        raise ValueError("linear weights and scales must be 2D")
    logical_k = packed.shape[1] * 2
    if x.shape[-1] != logical_k:
        raise ValueError(f"input K {x.shape[-1]} does not match weight K {logical_k}")
    if logical_k % 128 != 0:
        raise ValueError(f"logical K must be divisible by 128, got {logical_k}")
    if block_scales.shape != (packed.shape[0], logical_k // 32):
        raise ValueError("one E8M0 scale is required for every 32 MXFP4 values")

    input_shape = x.shape
    x_2d = x.reshape(-1, logical_k).contiguous()
    backend = _backend()
    if backend == "triton":
        _require_triton_compiler()
        output = torch.empty(
            (x_2d.shape[0], packed.shape[0]), dtype=x.dtype, device=x.device
        )
        block_m = 16
        block_n = 64
        grid = (
            triton.cdiv(x_2d.shape[0], block_m)
            * triton.cdiv(packed.shape[0], block_n),
        )
        _linear_mxfp4_bf16_kernel[grid](
            x_2d,
            packed,
            block_scales,
            output,
            x_2d.shape[0],
            packed.shape[0],
            logical_k,
            x_2d.stride(0),
            x_2d.stride(1),
            packed.stride(0),
            packed.stride(1),
            block_scales.stride(0),
            block_scales.stride(1),
            output.stride(0),
            output.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=128,
            num_warps=8,
            num_stages=3,
        )
    else:
        weight = dequant_mxfp4(packed, block_scales, x.dtype)
        output = torch.matmul(x_2d, weight.transpose(0, 1))
    if bias is not None:
        output = output + bias
    return output.reshape(*input_shape[:-1], packed.shape[0])


def fused_moe_mxfp4_soft(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    A_mx_scale: torch.Tensor | None,
    B_mx_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    swizzle_mx_a: bool,
    swizzle_mx_b: bool,
) -> None:
    """Correctness-first MXFP4 MoE fallback for architectures without scaled dot."""
    if swizzle_mx_a or swizzle_mx_b:
        raise NotImplementedError("the gfx936 software fallback does not support swizzled scales")
    if topk_ids.shape != topk_weights.shape or topk_ids.shape[1] != top_k:
        raise ValueError("top-k ids and weights must have matching (M, top_k) shapes")
    if B.dtype != torch.uint8 or B_mx_scale is None:
        raise TypeError("the gfx936 software fallback requires packed MXFP4 weights")

    matmul_dtype = C.dtype
    if matmul_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported software fallback compute dtype: {matmul_dtype}")

    backend = _backend()
    routing_ready = (
        A.dtype != torch.uint8
        and sorted_token_ids.numel() > 0
        and expert_ids.numel() > 0
        and num_tokens_post_padded.numel() == 1
        and B.shape[-1] * 2 % 128 == 0
        and sorted_token_ids.numel() % expert_ids.numel() == 0
    )
    if backend == "triton":
        _require_triton_compiler()
        if not routing_ready:
            raise RuntimeError(
                "the experimental Triton MoE backend requires unpacked "
                "activations and valid padded routing metadata"
            )
        block_m = sorted_token_ids.numel() // expert_ids.numel()
        if block_m not in (16, 32, 64, 128):
            raise RuntimeError(f"unsupported Triton routing block size: {block_m}")
        C.zero_()
        block_n = 64
        grid = (expert_ids.numel() * triton.cdiv(B.shape[1], block_n),)
        _fused_moe_mxfp4_bf16_kernel[grid](
            A,
            B,
            C,
            A_scale,
            B_scale,
            B_mx_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            B.shape[1],
            A.shape[1],
            topk_ids.numel(),
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(1),
            C.stride(2),
            B_mx_scale.stride(0),
            B_mx_scale.stride(1),
            B_mx_scale.stride(2),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=128,
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            num_warps=8,
            num_stages=3,
        )
        return

    del sorted_token_ids, expert_ids, num_tokens_post_padded

    if A.dtype == torch.uint8:
        if A_mx_scale is None:
            raise ValueError("packed MXFP4 activations require A_mx_scale")
        activations = dequant_mxfp4(A, A_mx_scale, matmul_dtype)
    else:
        if A_mx_scale is not None:
            raise ValueError("unpacked activations must not provide A_mx_scale")
        activations = A.to(matmul_dtype)
    C.zero_()
    global_scale = A_scale.reshape(-1)[0].to(torch.float32)
    for expert_index in range(B.shape[0]):
        positions = torch.nonzero(topk_ids == expert_index, as_tuple=False)
        if positions.numel() == 0:
            continue
        token_indices = positions[:, 0]
        slot_indices = positions[:, 1]
        expert_weights = dequant_mxfp4(
            B[expert_index], B_mx_scale[expert_index], matmul_dtype
        )
        output = torch.matmul(
            activations[token_indices],
            expert_weights.transpose(0, 1),
        )
        output = output * global_scale * B_scale[expert_index].to(torch.float32)
        if mul_routed_weight:
            output = output * topk_weights[token_indices, slot_indices, None]
        C[token_indices, slot_indices] = output.to(C.dtype)
