# Measured Qwen3-30B-A3B result

The sanitized result in this directory was measured on a Hygon BW100 system
reported as `gfx936`, using DTK 26.04, PyTorch 2.9, Triton 3.3, vLLM 0.15.1,
and tensor parallel size 2 for every compared path.

Model sources were `Qwen/Qwen3-30B-A3B` at revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` and
`nm-testing/Qwen3-30B-A3B-MXFP4A16`. The same-source audit confirmed matching
architecture, exact `lm_head`, and identical token IDs for every chat/PPL probe.

The formal hybrid checkpoint restored all routed expert gate/up/down matrices
to same-source BF16 and retained MXFP4 for non-expert weights. It passed all 11
predefined precision gates.

The passing implementation comparison is the explicit `reference` path versus
the guarded `triton` backend. The fused fast path uses `kpack=2` only for the
validated large-prefill shape range; small batches and router-sized outputs use
the safe BF16 matmul path. A pre-fix forced-fusion control is retained in the
result JSON to show the regression that the guard addresses.

The final same-input audit covered 4,032 dense calls on the two TP ranks with
zero mismatched calls or elements. Of those, 384 calls entered the fused fast
path and covered about 86.16% of compared output elements.

Important boundary: PPL used only 380 tokens. Full-sequence agreement against
official BF16 was 10/20 and was not a predefined hard gate. Treat this as a
reproducible implementation/precision probe, not proof of general model quality.
