#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare official BF16, hybrid reference, and a hybrid candidate result."""

import argparse
import json
import math
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def logprob_map(case):
    return {
        item["token_id"]: item["logprob"]
        for item in case["first_token_logprobs"]
    }


def compare_pair(left, right):
    left_cases = left["chat_cases"]
    right_cases = right["chat_cases"]
    if len(left_cases) != len(right_cases):
        raise ValueError("chat case counts differ")
    cases = []
    for left_case, right_case in zip(left_cases, right_cases):
        left_logprobs = logprob_map(left_case)
        right_logprobs = logprob_map(right_case)
        common = set(left_logprobs) & set(right_logprobs)
        cases.append(
            {
                "prompt_token_ids_equal": (
                    left_case["prompt_token_ids"]
                    == right_case["prompt_token_ids"]
                ),
                "first_token_equal": (
                    left_case["generated_token_ids"][0]
                    == right_case["generated_token_ids"][0]
                ),
                "full_sequence_equal": (
                    left_case["generated_token_ids"]
                    == right_case["generated_token_ids"]
                ),
                "top20_overlap": len(common),
                "max_common_logprob_abs_diff": max(
                    (
                        abs(left_logprobs[token] - right_logprobs[token])
                        for token in common
                    ),
                    default=math.inf,
                ),
            }
        )
    left_ppl = left["ppl"]["perplexity"]
    right_ppl = right["ppl"]["perplexity"]
    return {
        "case_count": len(cases),
        "first_token_equal_count": sum(c["first_token_equal"] for c in cases),
        "prompt_token_ids_equal_count": sum(
            c["prompt_token_ids_equal"] for c in cases
        ),
        "full_sequence_equal_count": sum(
            c["full_sequence_equal"] for c in cases
        ),
        "minimum_top20_overlap": min(c["top20_overlap"] for c in cases),
        "max_common_logprob_abs_diff": max(
            c["max_common_logprob_abs_diff"] for c in cases
        ),
        "left_perplexity": left_ppl,
        "right_perplexity": right_ppl,
        "relative_perplexity_delta": right_ppl / left_ppl - 1.0,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", required=True)
    parser.add_argument("--hybrid-reference", required=True)
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument("--hybrid-candidate")
    candidate.add_argument(
        "--hybrid-safe", dest="hybrid_candidate", help=argparse.SUPPRESS
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    official = load(args.official)
    hybrid_reference = load(args.hybrid_reference)
    hybrid_candidate = load(args.hybrid_candidate)
    checkpoint = load(args.checkpoint)

    quality = compare_pair(official, hybrid_reference)
    implementation = compare_pair(hybrid_reference, hybrid_candidate)
    same_tp = {
        official["tensor_parallel_size"],
        hybrid_reference["tensor_parallel_size"],
        hybrid_candidate["tensor_parallel_size"],
    }
    gates = {
        "checkpoint_restored_18432_exact_bf16": (
            checkpoint["restored_expert_weight_count"] == 18432
            and checkpoint["exact_validation_pass"]
        ),
        "checkpoint_no_quantized_expert_tensors": (
            checkpoint["remaining_expert_quantized_tensor_count"] == 0
        ),
        "checkpoint_nonexpert_mxfp4_preserved": (
            checkpoint["remaining_nonexpert_packed_weight_count"] == 240
        ),
        "checkpoint_ignore_pattern_active": (
            checkpoint["expert_ignore_pattern_present"]
            and checkpoint["reference_unquantized_moe_loaded"]
            and checkpoint.get(
                "candidate_unquantized_moe_loaded",
                checkpoint.get("safe_unquantized_moe_loaded", False),
            )
        ),
        "same_tensor_parallel_size_tp2": same_tp == {2},
        "implementation_full_sequence_exact": (
            implementation["full_sequence_equal_count"]
            == implementation["case_count"]
        ),
        "implementation_relative_perplexity_within_0_1pct": (
            abs(implementation["relative_perplexity_delta"]) <= 0.001
        ),
        "implementation_logprob_close": (
            implementation["max_common_logprob_abs_diff"] <= 1e-5
        ),
        "quality_relative_perplexity_within_5pct": (
            quality["relative_perplexity_delta"] <= 0.05
        ),
        "quality_chat_tokenization_exact": (
            quality["prompt_token_ids_equal_count"] == quality["case_count"]
        ),
        "quality_first_token_agreement_at_least_90pct": (
            quality["first_token_equal_count"] / quality["case_count"] >= 0.90
        ),
    }
    payload = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "passed_gate_count": sum(gates.values()),
        "total_gate_count": len(gates),
        "gates": gates,
        "checkpoint": checkpoint,
        "candidate_mode": hybrid_candidate["mode"],
        "quality_official_bf16_vs_hybrid_reference": quality,
        "implementation_hybrid_reference_vs_candidate": implementation,
    }
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "passed": payload["passed_gate_count"],
                "total": payload["total_gate_count"],
            }
        )
    )
    raise SystemExit(0 if payload["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
