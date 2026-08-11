# Qwen3-30B-A3B precision-gate example

This example separates three questions:

1. Does the hybrid checkpoint have the intended structure?
2. Does the gfx936 safe path match an independent dequantized reference?
3. Does the hybrid checkpoint remain within the predefined quality thresholds
   relative to the same-source BF16 model at the same tensor-parallel size?

## Inputs

- a Qwen3-30B-A3B BF16 checkpoint;
- its same-source OCP MXFP4 checkpoint;
- vLLM 0.15.1 with the adapter in this repository;
- two `gfx936` devices for the reproduced TP=2 gate.

The measured example used `Qwen/Qwen3-30B-A3B` revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` and
`nm-testing/Qwen3-30B-A3B-MXFP4A16`.

The script assumes the Qwen3 architecture used by the measured checkpoint:
48 layers, 128 routed experts per layer, and gate/up/down shapes encoded in
`build_all_experts_bf16_checkpoint.py`. Fail-loud shape checks prevent applying
it silently to a different architecture.

## 1. Build the hybrid checkpoint

```bash
QUANT_MODEL=/models/Qwen3-30B-A3B-MXFP4A16 \
BF16_MODEL=/models/Qwen3-30B-A3B-BF16 \
OUTPUT_MODEL=/models/Qwen3-30B-A3B-MXFP4A16-expertsBF16 \
python build_all_experts_bf16_checkpoint.py
```

The builder performs exact tensor validation before atomically renaming the
staging directory to the requested output.

## 2. Run the three paths at TP=2

`eval_qwen3_30b_moe_quality.py` emits raw JSON for 20 deterministic chat probes
and a 380-token PPL probe.

- official BF16: `MXFP4_QUALITY_MODE=base`;
- hybrid independent dequant reference: set `MXFP4_QUALITY_MODE=reference` and
  name its output `hybrid_reference_tp2.json`;
- hybrid gfx936 safe path: set `MXFP4_QUALITY_MODE=safe`, import
  `integration.vllm.vllm_mxfp4_gfx936` before constructing `vllm.LLM` and name
  its output `hybrid_safe_tp2.json`.

Keep `MXFP4_TP_SIZE=2`, prompts, tokenizer, seed, and model revisions fixed
across all paths.

## 3. Evaluate the gate

Create a checkpoint-summary JSON with the fields shown in
`checkpoint_summary.example.json`, then run:

```bash
python compare_gate.py \
  --official official_bf16_tp2.json \
  --hybrid-reference hybrid_reference_tp2.json \
  --hybrid-safe hybrid_safe_tp2.json \
  --checkpoint checkpoint_summary.json \
  --output formal_experts_bf16_gate.json
```

The formal measured result is preserved under
`../../results/qwen3_30b_a3b/`. It passed 11/11 predefined gates. The probe is
deliberately narrow and is not a substitute for a standard benchmark suite.
