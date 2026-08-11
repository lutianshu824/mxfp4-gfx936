#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


QUANT = Path(os.getenv("QUANT_MODEL", "/models/Qwen3-30B-A3B-MXFP4A16"))
BF16 = Path(os.getenv("BF16_MODEL", "/models/Qwen3-30B-A3B-BF16"))
OUTPUT = Path(
    os.getenv(
        "OUTPUT_MODEL", "/models/Qwen3-30B-A3B-MXFP4A16-expertsBF16"
    )
)
STAGING = OUTPUT.with_name(f"{OUTPUT.name}.building")
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
MODULES = [
    f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
    for layer in range(48)
    for expert in range(128)
    for projection in PROJECTIONS
]
EXPERT_GROUPS = [f"model.layers.{layer}.mlp.experts" for layer in range(48)]
EXPERT_IGNORE_PATTERN = (
    r"re:.*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"
)


def load_index(root):
    return json.loads((root / "model.safetensors.index.json").read_text())


def tensor_nbytes(tensor):
    return tensor.numel() * tensor.element_size()


def expected_shape(module):
    if module.endswith("down_proj"):
        return (2048, 768)
    return (768, 2048)


def copy_metadata_files():
    STAGING.mkdir(parents=True)
    excluded = {"config.json", "model.safetensors.index.json"}
    for source in QUANT.iterdir():
        if (
            not source.is_file()
            or source.name in excluded
            or source.name.endswith(".safetensors")
        ):
            continue
        shutil.copy2(source, STAGING / source.name)


def build():
    if OUTPUT.exists() or STAGING.exists():
        raise SystemExit(f"output already exists: {OUTPUT} or {STAGING}")

    quant_index = load_index(QUANT)
    bf16_index = load_index(BF16)
    quant_map = quant_index["weight_map"]
    bf16_map = bf16_index["weight_map"]
    output_map = dict(quant_map)
    copy_metadata_files()

    removed_bytes = 0
    added_bytes = 0
    restored_count = 0
    shard_names = sorted(set(quant_map.values()))
    for shard_number, shard_name in enumerate(shard_names, start=1):
        with safe_open(
            QUANT / shard_name, framework="pt", device="cpu"
        ) as source:
            metadata = source.metadata()
            tensors = {
                name: source.get_tensor(name).contiguous()
                for name in source.keys()
            }

        shard_modules = [
            module
            for module in MODULES
            if quant_map[f"{module}.weight_packed"] == shard_name
        ]
        for module in shard_modules:
            for name in (
                f"{module}.weight_packed",
                f"{module}.weight_scale",
            ):
                tensor = tensors.pop(name)
                removed_bytes += tensor_nbytes(tensor)
                del output_map[name]

        bf16_names_by_shard = {}
        for module in shard_modules:
            weight_name = f"{module}.weight"
            bf16_names_by_shard.setdefault(bf16_map[weight_name], []).append(
                weight_name
            )
        for bf16_shard, weight_names in bf16_names_by_shard.items():
            with safe_open(
                BF16 / bf16_shard, framework="pt", device="cpu"
            ) as source:
                for weight_name in weight_names:
                    weight = source.get_tensor(weight_name).contiguous()
                    module = weight_name.removesuffix(".weight")
                    if (
                        weight.dtype != torch.bfloat16
                        or tuple(weight.shape) != expected_shape(module)
                    ):
                        raise RuntimeError(
                            f"unexpected {weight_name}: dtype={weight.dtype} "
                            f"shape={tuple(weight.shape)}"
                        )
                    tensors[weight_name] = weight
                    output_map[weight_name] = shard_name
                    added_bytes += tensor_nbytes(weight)
                    restored_count += 1

        print(
            f"SAVE_SHARD {shard_number}/{len(shard_names)} {shard_name} "
            f"restored={len(shard_modules)} tensors={len(tensors)}",
            flush=True,
        )
        save_file(tensors, STAGING / shard_name, metadata=metadata)
        del tensors

    if restored_count != len(MODULES):
        raise RuntimeError(
            f"expected {len(MODULES)} restored weights, got {restored_count}"
        )

    config = json.loads((QUANT / "config.json").read_text())
    quantization_config = config["quantization_config"]
    ignore = list(quantization_config.get("ignore") or [])
    for group in EXPERT_GROUPS:
        if group not in ignore:
            ignore.append(group)
    if EXPERT_IGNORE_PATTERN not in ignore:
        ignore.append(EXPERT_IGNORE_PATTERN)
    quantization_config["ignore"] = ignore
    (STAGING / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    )

    quant_index["weight_map"] = output_map
    metadata = quant_index.setdefault("metadata", {})
    if "total_size" in metadata:
        metadata["total_size"] += added_bytes - removed_bytes
    (STAGING / "model.safetensors.index.json").write_text(
        json.dumps(quant_index, ensure_ascii=False, indent=2) + "\n"
    )

    provenance = {
        "base_quant_model": str(QUANT),
        "bf16_source": str(BF16),
        "restored_scope": "all routed experts gate_proj/up_proj/down_proj",
        "restored_module_count": len(MODULES),
        "ignored_fused_expert_groups": EXPERT_GROUPS,
        "expert_ignore_pattern": EXPERT_IGNORE_PATTERN,
        "removed_quantized_bytes": removed_bytes,
        "added_bf16_bytes": added_bytes,
        "single_change": (
            "restore all routed expert gate/up/down weights to official BF16; "
            "keep every non-expert weight from the MXFP4 checkpoint"
        ),
    }
    (STAGING / "experts_bf16_checkpoint.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
    )

    validate_structure(output_map, bf16_map)
    validate_exact_weights(output_map, bf16_map)
    STAGING.rename(OUTPUT)
    print(f"BUILD_COMPLETE {OUTPUT}", flush=True)


def validate_structure(output_map, bf16_map):
    written_map = json.loads(
        (STAGING / "model.safetensors.index.json").read_text()
    )["weight_map"]
    if written_map != output_map:
        raise RuntimeError("written weight map differs from constructed map")

    config = json.loads((STAGING / "config.json").read_text())
    ignore = set(config["quantization_config"]["ignore"])
    if not set(EXPERT_GROUPS).issubset(ignore):
        raise RuntimeError("one or more fused expert groups are missing from ignore")
    if EXPERT_IGNORE_PATTERN not in ignore:
        raise RuntimeError("expert projection regex is missing from ignore")

    for module in MODULES:
        weight_name = f"{module}.weight"
        if weight_name not in written_map or weight_name not in bf16_map:
            raise RuntimeError(f"missing restored expert weight: {weight_name}")
        if (
            f"{module}.weight_packed" in written_map
            or f"{module}.weight_scale" in written_map
        ):
            raise RuntimeError(f"quantized expert keys remain: {module}")

    remaining_packed = [
        name for name in written_map if name.endswith(".weight_packed")
    ]
    if not remaining_packed:
        raise RuntimeError("no non-expert MXFP4 weights remain")
    if any(".mlp.experts." in name for name in remaining_packed):
        raise RuntimeError("an expert MXFP4 packed tensor remains")
    print(
        f"STRUCTURE_VALIDATION_PASS restored={len(MODULES)} "
        f"ignored_groups={len(EXPERT_GROUPS)} "
        f"remaining_nonexpert_packed={len(remaining_packed)}",
        flush=True,
    )


def validate_exact_weights(output_map, bf16_map):
    exact = 0
    output_shards = sorted(set(output_map[f"{module}.weight"] for module in MODULES))
    for shard_number, output_shard in enumerate(output_shards, start=1):
        shard_names = [
            f"{module}.weight"
            for module in MODULES
            if output_map[f"{module}.weight"] == output_shard
        ]
        names_by_bf16_shard = {}
        for name in shard_names:
            names_by_bf16_shard.setdefault(bf16_map[name], []).append(name)
        with safe_open(
            STAGING / output_shard, framework="pt", device="cpu"
        ) as restored:
            for bf16_shard, names in names_by_bf16_shard.items():
                with safe_open(
                    BF16 / bf16_shard, framework="pt", device="cpu"
                ) as original:
                    for name in names:
                        if not torch.equal(
                            restored.get_tensor(name), original.get_tensor(name)
                        ):
                            raise RuntimeError(
                                f"restored expert differs from source: {name}"
                            )
                        exact += 1
        print(
            f"VERIFY_SHARD {shard_number}/{len(output_shards)} "
            f"{output_shard} exact={len(shard_names)}",
            flush=True,
        )
    if exact != len(MODULES):
        raise RuntimeError(f"expected {len(MODULES)} exact tensors, got {exact}")
    print(f"EXACT_VALIDATION_PASS exact_bf16={exact}", flush=True)


if __name__ == "__main__":
    build()
