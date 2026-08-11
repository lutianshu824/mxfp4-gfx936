# SPDX-License-Identifier: MIT

import importlib.util
import os
from pathlib import Path

import pytest
import torch

def _load_soft_module():
    default = (
        Path(__file__).resolve().parents[1]
        / "src/aiter/ops/triton/moe_op_mxfp4_soft.py"
    )
    path = Path(os.getenv("MXFP4_SOFT_MODULE", default))
    spec = importlib.util.spec_from_file_location("moe_op_mxfp4_soft", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


soft = _load_soft_module()
_MXFP4_E2M1_VALUES = soft._MXFP4_E2M1_VALUES
dequant_mxfp4 = soft.dequant_mxfp4
fused_moe_mxfp4_soft = soft.fused_moe_mxfp4_soft


def torch_dynamic_mxfp4_quant(x):
    """Deterministic OCP E2M1/E8M0 test quantizer derived from AITER tests."""
    block_size = 32
    if x.shape[-1] % block_size:
        raise ValueError("test quantizer requires K divisible by 32")
    original_shape = x.shape
    blocks = x.reshape(-1, x.shape[-1] // block_size, block_size).float()
    amax = torch.max(torch.abs(blocks), dim=-1).values
    amax = amax.view(torch.int32)
    amax = ((amax + 0x200000) & 0xFF800000).view(torch.float32)
    scale_exponent = torch.clamp(torch.log2(amax).floor() - 2, -127, 127)
    quantized = (blocks * torch.exp2(-scale_exponent).unsqueeze(-1)).view(
        torch.int32
    )
    signs = quantized & 0x80000000
    exponents = (quantized >> 23) & 0xFF
    mantissas = quantized & 0x7FFFFF
    adjusted = 127 - exponents - 1
    mantissas = torch.where(
        exponents < 127,
        (0x400000 | (mantissas >> 1)) >> adjusted,
        mantissas,
    )
    exponents = torch.where(exponents > 126, exponents, 126) - 126
    combined = (((exponents << 2) | (mantissas >> 21)) + 1) >> 1
    magnitude = torch.where(combined < 7, combined, 7)
    codes = (((signs >> 28) & 0xF) | magnitude).to(torch.uint8)
    packed = codes[..., ::2] | (codes[..., 1::2] << 4)
    packed = packed.flatten(-2).reshape(*original_shape[:-1], original_shape[-1] // 2)
    scales = (scale_exponent.to(torch.uint8) + 127).reshape(
        *original_shape[:-1], original_shape[-1] // block_size
    )
    return packed, scales


def test_safe_backend_is_default(monkeypatch):
    monkeypatch.delenv("MXFP4_GFX936_BACKEND", raising=False)
    assert soft._backend() == "safe"


def test_unknown_backend_fails_loud(monkeypatch):
    monkeypatch.setenv("MXFP4_GFX936_BACKEND", "unknown")
    with pytest.raises(ValueError, match="unknown gfx936 MXFP4 backend"):
        soft._backend()


def test_triton_backend_requires_compiler(monkeypatch):
    monkeypatch.setenv("MXFP4_GFX936_BACKEND", "triton")
    monkeypatch.setattr(soft, "_has_compatible_triton_compiler", lambda: False)
    with pytest.raises(RuntimeError, match="TRITON_HIP_CLANG_PATH"):
        soft._require_triton_compiler()


def test_dequant_all_e2m1_codes():
    packed = torch.tensor(
        [[code | ((code + 1) << 4) for code in range(0, 16, 2)] * 2],
        device="cuda",
        dtype=torch.uint8,
    )
    scales = torch.full((1, 1), 127, device="cuda", dtype=torch.uint8)
    actual = dequant_mxfp4(packed, scales)
    expected = torch.tensor(
        [_MXFP4_E2M1_VALUES], device="cuda", dtype=torch.float32
    ).repeat(1, 2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("m_value", [1, 8])
def test_linear_safe_matches_explicit_dequant(m_value, monkeypatch):
    monkeypatch.setenv("MXFP4_GFX936_BACKEND", "safe")
    torch.manual_seed(20260811 + m_value)
    n_value, k_value = 64, 128
    activations = torch.randn(
        (m_value, k_value), device="cuda", dtype=torch.bfloat16
    )
    weights = torch.randn(
        (n_value, k_value), device="cuda", dtype=torch.bfloat16
    )
    packed, scales = torch_dynamic_mxfp4_quant(weights)
    expected = activations @ dequant_mxfp4(
        packed, scales, torch.bfloat16
    ).T
    actual = soft.linear_mxfp4_soft(activations, packed, scales)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("m_value", [1, 2, 4, 8, 64])
@pytest.mark.parametrize("quantize_activations", [False, True])
@pytest.mark.parametrize("mul_routed_weight", [False, True])
def test_fused_moe_mxfp4_soft(m_value, quantize_activations, mul_routed_weight):
    torch.manual_seed(20260810 + m_value)
    n_value, k_value, experts, top_k = 64, 128, 4, 2
    activations = torch.randn(
        (m_value, k_value), device="cuda", dtype=torch.bfloat16
    )
    weights = torch.randn(
        (experts, n_value, k_value), device="cuda", dtype=torch.bfloat16
    )
    packed_weights, weight_scales = torch_dynamic_mxfp4_quant(weights)
    dequant_weights = dequant_mxfp4(
        packed_weights, weight_scales, torch.bfloat16
    )

    if quantize_activations:
        kernel_activations, activation_scales = torch_dynamic_mxfp4_quant(activations)
        reference_activations = dequant_mxfp4(
            kernel_activations, activation_scales, torch.bfloat16
        )
    else:
        kernel_activations, activation_scales = activations, None
        reference_activations = activations

    logits = torch.randn((m_value, experts), device="cuda", dtype=torch.float32)
    probabilities = torch.softmax(logits, dim=-1)
    topk_weights, topk_ids = torch.topk(probabilities, k=top_k, dim=-1)
    actual = torch.empty(
        (m_value, top_k, n_value), device="cuda", dtype=torch.bfloat16
    )
    scalar_a = torch.tensor([0.75], device="cuda", dtype=torch.float32)
    scalar_b = torch.linspace(0.5, 1.25, experts, device="cuda")

    fused_moe_mxfp4_soft(
        kernel_activations,
        packed_weights,
        actual,
        scalar_a,
        scalar_b,
        activation_scales,
        weight_scales,
        topk_weights,
        topk_ids,
        torch.empty(0, device="cuda"),
        torch.empty(0, device="cuda"),
        torch.empty(0, device="cuda"),
        mul_routed_weight,
        top_k,
        False,
        False,
    )

    expected = torch.empty_like(actual)
    for token_index in range(m_value):
        for slot_index in range(top_k):
            expert_index = int(topk_ids[token_index, slot_index])
            value = torch.matmul(
                reference_activations[token_index],
                dequant_weights[expert_index].transpose(0, 1),
            )
            value = value * scalar_a[0] * scalar_b[expert_index]
            if mul_routed_weight:
                value = value * topk_weights[token_index, slot_index]
            expected[token_index, slot_index] = value.to(expected.dtype)

    # The fallback batches tokens by expert, while this reference uses GEMV
    # per token. BF16 reduction order can therefore differ by one ULP.
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
