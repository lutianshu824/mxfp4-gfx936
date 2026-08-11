# SPDX-License-Identifier: MIT
"""Minimal vLLM 0.15 adapter for the gfx936 MXFP4 software kernels.

Import this module before constructing ``vllm.LLM``. This adapter covers the
compressed-tensors W4A16 dense and W4A4 MXFP4 MoE methods used by the validated
Qwen3 checkpoint. It is a correctness-first runtime integration, not a stable
vLLM plugin API.
"""

import importlib.util
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import vllm
from compressed_tensors.compressors.quantized_compressors.fp4_quantized import (
    unpack_fp4_from_uint8,
)
from packaging.version import Version
from torch.nn.parameter import Parameter


if Version(vllm.__version__.split("+")[0]) < Version("0.15.0"):
    raise RuntimeError(
        f"the gfx936 MXFP4 adapter requires vLLM >= 0.15, got {vllm.__version__}"
    )

_IMPLEMENTATION = os.getenv("MXFP4_IMPLEMENTATION", "safe")
if _IMPLEMENTATION == "bf16":
    _IMPLEMENTATION = "reference"
if _IMPLEMENTATION not in ("reference", "safe", "triton"):
    raise ValueError(f"unknown MXFP4 implementation: {_IMPLEMENTATION}")
os.environ["MXFP4_GFX936_BACKEND"] = (
    "triton" if _IMPLEMENTATION == "triton" else "safe"
)


def _load_soft_module():
    default = (
        Path(__file__).resolve().parents[2]
        / "src/aiter/ops/triton/moe_op_mxfp4_soft.py"
    )
    path = Path(os.getenv("MXFP4_SOFT_MODULE", default))
    if not path.is_file():
        raise FileNotFoundError(f"MXFP4 soft module not found: {path}")
    spec = importlib.util.spec_from_file_location("moe_op_mxfp4_soft", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load MXFP4 soft module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


soft = _load_soft_module()
_official_unpack_fp4 = getattr(
    unpack_fp4_from_uint8, "__wrapped__", unpack_fp4_from_uint8
)

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E402
    CompressedTensorsW4A4Mxfp4MoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a16_mxfp4 import (  # noqa: E402
    CompressedTensorsW4A16Mxfp4,
)


def _dense_process_weights_after_loading(self, layer):
    layer.weight_packed = Parameter(layer.weight_packed.data, requires_grad=False)
    layer.weight_scale = Parameter(layer.weight_scale.data, requires_grad=False)


def _reference_dequant(packed, block_scales, dtype):
    """Independent reference using compressed-tensors unpack + torch.ldexp."""
    logical_k = packed.shape[-1] * 2
    values = _official_unpack_fp4(
        packed,
        packed.shape[-2],
        logical_k,
        dtype=torch.float32,
    )
    exponents = block_scales.to(torch.int32) - 127
    scales = torch.ldexp(
        torch.ones_like(exponents, dtype=torch.float32), exponents
    )
    scales = torch.where(
        block_scales == 255,
        torch.full_like(scales, float("nan")),
        scales,
    )
    return (values * scales.repeat_interleave(32, dim=-1)).to(dtype)


def _bf16_linear(x, packed, block_scales, bias=None):
    input_shape = x.shape
    logical_k = packed.shape[-1] * 2
    x_2d = x.reshape(-1, logical_k)
    weight = _reference_dequant(packed, block_scales, x.dtype)
    output = x_2d @ weight.T
    if bias is not None:
        output = output + bias
    return output.reshape(*input_shape[:-1], packed.shape[0])


def _dense_apply_weights(
    self,
    layer,
    x,
    bias=None,
    input_quant_args=None,
    silu_quant_args=None,
):
    del self, input_quant_args, silu_quant_args
    if _IMPLEMENTATION == "reference":
        return _bf16_linear(x, layer.weight_packed, layer.weight_scale, bias)
    return soft.linear_mxfp4_soft(x, layer.weight_packed, layer.weight_scale, bias)


def _make_routing(topk_ids, block_m=64):
    flat_ids = topk_ids.flatten()
    route_count = flat_ids.numel()
    sorted_blocks = []
    expert_blocks = []
    for expert in torch.unique(flat_ids, sorted=True).tolist():
        route_ids = torch.nonzero(flat_ids == expert, as_tuple=False).flatten()
        for start in range(0, route_ids.numel(), block_m):
            chunk = route_ids[start : start + block_m]
            padded = torch.full(
                (block_m,),
                route_count,
                dtype=torch.int32,
                device=topk_ids.device,
            )
            padded[: chunk.numel()] = chunk.to(torch.int32)
            sorted_blocks.append(padded)
            expert_blocks.append(expert)
    if not sorted_blocks:
        raise ValueError("topk_ids must contain at least one routed token")
    sorted_ids = torch.cat(sorted_blocks)
    expert_ids = torch.tensor(
        expert_blocks, dtype=torch.int32, device=topk_ids.device
    )
    post_padded = torch.tensor(
        [sorted_ids.numel()], dtype=torch.int32, device=topk_ids.device
    )
    return sorted_ids, expert_ids, post_padded


def _moe_gemm(
    activations,
    packed,
    scales,
    topk_ids,
    topk_weights,
    multiply_route_weights,
):
    sorted_ids, expert_ids, post_padded = _make_routing(topk_ids)
    output = torch.empty(
        (topk_ids.shape[0], topk_ids.shape[1], packed.shape[1]),
        dtype=activations.dtype,
        device=activations.device,
    )
    a_scale = torch.ones(1, dtype=torch.float32, device=activations.device)
    b_scale = torch.ones(
        packed.shape[0], dtype=torch.float32, device=activations.device
    )
    soft.fused_moe_mxfp4_soft(
        activations,
        packed,
        output,
        a_scale,
        b_scale,
        None,
        scales,
        topk_weights,
        topk_ids,
        sorted_ids,
        expert_ids,
        post_padded,
        multiply_route_weights,
        topk_ids.shape[1],
        False,
        False,
    )
    return output


def _bf16_moe_gemm(
    activations,
    packed,
    scales,
    topk_ids,
    topk_weights,
    multiply_route_weights,
):
    output = torch.zeros(
        (topk_ids.shape[0], topk_ids.shape[1], packed.shape[1]),
        dtype=activations.dtype,
        device=activations.device,
    )
    for expert in torch.unique(topk_ids, sorted=True).tolist():
        positions = torch.nonzero(topk_ids == expert, as_tuple=False)
        token_ids = positions[:, 0]
        slot_ids = positions[:, 1]
        weight = _reference_dequant(
            packed[expert], scales[expert], activations.dtype
        )
        values = activations[token_ids] @ weight.T
        if multiply_route_weights:
            values = values * topk_weights[token_ids, slot_ids, None]
        output[token_ids, slot_ids] = values.to(activations.dtype)
    return output


def _moe_process_weights_after_loading(self, layer):
    del self
    layer.w13_weight = Parameter(
        layer.w13_weight_packed.data, requires_grad=False
    )
    layer.w13_weight_scale = Parameter(
        layer.w13_weight_scale.data, requires_grad=False
    )
    layer.w2_weight = Parameter(
        layer.w2_weight_packed.data, requires_grad=False
    )
    layer.w2_weight_scale = Parameter(
        layer.w2_weight_scale.data, requires_grad=False
    )
    delattr(layer, "w13_weight_packed")
    delattr(layer, "w2_weight_packed")


def _moe_apply(self, layer, x, topk_weights, topk_ids, use_nn_moe=False):
    del self, use_nn_moe
    if layer.activation != "silu":
        raise NotImplementedError(
            f"the gfx936 MXFP4 adapter supports silu, got {layer.activation}"
        )
    moe_gemm = _bf16_moe_gemm if _IMPLEMENTATION == "reference" else _moe_gemm
    gate_up = moe_gemm(
        x,
        layer.w13_weight,
        layer.w13_weight_scale,
        topk_ids,
        topk_weights,
        False,
    )
    intermediate_size = gate_up.shape[-1] // 2
    routed = (
        F.silu(gate_up[..., :intermediate_size])
        * gate_up[..., intermediate_size:]
    ).reshape(topk_ids.numel(), intermediate_size)
    down_ids = topk_ids.reshape(topk_ids.numel(), 1)
    down_weights = topk_weights.reshape(topk_ids.numel(), 1)
    down = moe_gemm(
        routed,
        layer.w2_weight,
        layer.w2_weight_scale,
        down_ids,
        down_weights,
        True,
    )
    return down[:, 0].reshape(
        topk_ids.shape[0], topk_ids.shape[1], x.shape[-1]
    ).sum(dim=1)


CompressedTensorsW4A16Mxfp4.process_weights_after_loading = (
    _dense_process_weights_after_loading
)
CompressedTensorsW4A16Mxfp4.apply_weights = _dense_apply_weights
CompressedTensorsW4A4Mxfp4MoEMethod.process_weights_after_loading = (
    _moe_process_weights_after_loading
)
CompressedTensorsW4A4Mxfp4MoEMethod.apply = _moe_apply

print(
    f"MXFP4_GFX936_ADAPTER_ACTIVE vllm={vllm.__version__} "
    f"implementation={_IMPLEMENTATION}"
)
