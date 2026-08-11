# MXFP4 software fallback for gfx936

Correctness-first OCP MXFP4 inference on Hygon DCU `gfx936`, with AITER and
vLLM integration plus a Qwen3-30B-A3B precision-gate example.

中文简介：本仓库公开在 `gfx936` 上实现 MXFP4 的软件回退方法，以及
Qwen3-30B-A3B “routed experts 使用 BF16、其余权重保持 MXFP4”的同 TP
精度验证示例。仓库不包含模型权重。

## Why this exists

The native AITER MXFP4 MoE kernel uses Triton's `tt.dot_scaled` with E2M1
operands. On the tested `gfx936` toolchain, compilation stops at
`convert-triton-amdgpu-to-llvm` with `Unsupported DotScaleOp`. Adding `gfx936`
to an architecture allow-list does not provide the missing compiler lowering.

The precision-safe backend keeps checkpoint weights packed, explicitly
dequantizes OCP MXFP4 on device, and calls the platform BF16 matmul. An
experimental fused backend performs the following work inside a Triton tile:

1. unpack the low/high OCP E2M1 nibbles in original K order;
2. apply one E8M0 scale per 32 values;
3. interleave the decoded values into BF16 tiles;
4. execute a single BF16 dot and accumulate in FP32.

The fused backend uses a single dot because two separate low/high dots produced small
rounding differences that could be amplified by MoE routing.

The backends are explicit:

- `safe` is the default and the only backend covered by the passing Qwen3 gate;
- `triton` is experimental and must be requested explicitly. It fails loudly
  when the compiler path is unavailable and is not presented as precision-passed.

## Scope

| Capability | Status |
|---|---|
| BF16/FP16 activation × packed MXFP4 weight | Supported |
| Packed MXFP4 activation × packed MXFP4 weight | Correctness fallback supported |
| Dense W4A16 linear, safe backend | Supported and precision-gated |
| Routed MoE with route weights, safe backend | Supported |
| Fused Triton dense/MoE | Experimental; not precision-released |
| E2M1 values + E8M0 scale, block size 32 | Supported |
| Swizzled MX scales | Not supported; fails explicitly |
| Native MXFP4 throughput | Not claimed |

This is a software fallback. It removes the `DotScaleOp` blocker but does not
turn `gfx936` into hardware-native MXFP4 execution.

## Repository layout

- `src/aiter/ops/triton/moe_op_mxfp4_soft.py`: safe fallback plus experimental
  dense and routed-MoE Triton kernels.
- `integration/aiter/moe_op_mxfp4.py`: AITER entry file with the `gfx936`
  dispatch. It is based on OpenDAS AITER commit `e99c6755`.
- `integration/vllm/vllm_mxfp4_gfx936.py`: minimal vLLM 0.15 runtime adapter
  for compressed-tensors W4A16 and W4A4 MoE schemes.
- `tests/test_moe_mxfp4_soft.py`: 21-case operator gate.
- `examples/qwen3_30b_a3b/`: checkpoint builder and same-TP quality gate.
- `results/qwen3_30b_a3b/`: sanitized measured results.

## AITER integration

Start from OpenDAS AITER commit
`e99c675538b77b1cc99917546aced406c0ac7ccb`, then copy the two implementation
files into the matching source tree:

```bash
cp src/aiter/ops/triton/moe_op_mxfp4_soft.py \
  <aiter>/aiter/ops/triton/moe_op_mxfp4_soft.py
cp integration/aiter/moe_op_mxfp4.py \
  <aiter>/aiter/ops/triton/moe_op_mxfp4.py
cp tests/test_moe_mxfp4_soft.py \
  <aiter>/op_tests/triton_tests/test_moe_mxfp4_soft.py
```

Run inside a DCU environment with AITER's test dependencies:

```bash
cd <aiter>
pytest -q op_tests/triton_tests/test_moe_mxfp4_soft.py
```

The original operator matrix passed 21/21 cases: all 16 E2M1 codes, M values
1/2/4/8/64, BF16 and MXFP4 activations, and routed-weight on/off. The public
test file adds backend-selection, fail-loud compiler, and dense-safe exactness
checks; the release suite passes 26/26.

## vLLM integration

The adapter is intentionally a runtime patch so the integration boundary stays
visible. `safe` is the default. Point it at the soft-kernel module and import it before constructing
`vllm.LLM`:

```bash
export MXFP4_SOFT_MODULE="$PWD/src/aiter/ops/triton/moe_op_mxfp4_soft.py"
export MXFP4_IMPLEMENTATION=safe
python -c 'import integration.vllm.vllm_mxfp4_gfx936; import your_server'
```

Tested end-to-end stack:

- Hygon BW100, reported architecture `gfx936`
- DTK 26.04
- PyTorch 2.9
- Triton 3.3 from the validated vLLM image
- vLLM 0.15.1

Only the experimental fused backend needs `MXFP4_IMPLEMENTATION=triton` and an
explicit `TRITON_HIP_CLANG_PATH`. It raises an error instead of silently falling
back when the compiler is unavailable.

## Qwen3-30B-A3B example

The original all-MXFP4 checkpoint failed the predefined quality gate even
though the software implementation matched an independent dequantized reference.
The release audit later established that this passing path was the safe BF16
fallback, not the fused Triton backend.
Single-factor ablations identified routed expert gate/up as the largest loss
source. The final example therefore restores all routed expert gate/up/down
weights from the same-source BF16 checkpoint and keeps non-expert weights in
MXFP4.

Build the hybrid checkpoint:

```bash
export QUANT_MODEL=/models/Qwen3-30B-A3B-MXFP4A16
export BF16_MODEL=/models/Qwen3-30B-A3B-BF16
export OUTPUT_MODEL=/models/Qwen3-30B-A3B-MXFP4A16-expertsBF16
python examples/qwen3_30b_a3b/build_all_experts_bf16_checkpoint.py
```

The builder is fail-loud: it refuses to overwrite an output, verifies all
18,432 restored expert tensors exactly against the BF16 source, removes all
expert packed/scale tensors, and confirms that non-expert MXFP4 tensors remain.

Evaluate the official BF16 model and the hybrid model with the same TP size.
The example result used TP=2:

```bash
MXFP4_QUALITY_MODE=base MXFP4_TP_SIZE=2 \
MXFP4_MODEL=/models/Qwen3-30B-A3B-BF16 \
MXFP4_RESULT_PATH=official_bf16_tp2.json \
python examples/qwen3_30b_a3b/eval_qwen3_30b_moe_quality.py

MXFP4_QUALITY_MODE=safe MXFP4_TP_SIZE=2 \
MXFP4_MODEL=/models/Qwen3-30B-A3B-MXFP4A16-expertsBF16 \
MXFP4_RESULT_PATH=hybrid_safe_tp2.json \
python examples/qwen3_30b_a3b/eval_qwen3_30b_moe_quality.py
```

See `examples/qwen3_30b_a3b/README.md` for the independent-dequant comparison
and gate command.

## Measured result

The formal hybrid checkpoint passed all 11 predefined same-TP precision gates:

- implementation: independent reference versus safe backend, with 20/20 full
  token sequences and top-20 logprobs exact;
- PPL: official BF16 `58.3569`, hybrid reference/safe `56.9245`;
- quality: 20/20 tokenization and 18/20 first-token agreement;
- structure: 18,432 exact BF16 expert tensors, zero quantized expert tensors,
  and 240 non-expert packed MXFP4 weights retained.

This is a 380-token probe, not a broad benchmark. Full-sequence agreement with
official BF16 was 10/20 and was not a predefined hard gate. The result supports
implementation correctness and the stated narrow precision gate; it does not
establish general quality superiority. The hybrid checkpoint also grows from
about 17 GiB to about 56 GiB, so throughput and capacity acceptance remain.

A forced fused-Triton control run produced PPL `57.7941` versus reference
`56.9245` (+1.5276%), with 19/20 full sequences equal. It failed the predefined
0.1% implementation threshold. The repository therefore exposes that backend
only as experimental and does not use it for the passing Qwen3 example.

Full sanitized evidence is in
`results/qwen3_30b_a3b/formal_experts_bf16_gate.json`.

## License and attribution

MIT. The AITER-derived files retain their SPDX header. See `NOTICE` and
`LICENSE`. Model weights are governed by their own upstream licenses and are
not included here.
