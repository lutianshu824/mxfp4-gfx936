#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

import json
import math
import os
import time
from pathlib import Path


MODE = os.getenv("MXFP4_QUALITY_MODE", "base")
if MODE == "bf16":
    MODE = "reference"
if MODE not in ("base", "reference", "safe", "triton"):
    raise ValueError(f"unknown quality mode: {MODE}")
if MODE != "base":
    os.environ["MXFP4_IMPLEMENTATION"] = MODE
    import integration.vllm.vllm_mxfp4_gfx936  # noqa: F401

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


CHAT_PROMPTS = [
    "计算 17×23，只给出结果。",
    "把“海光支持高性能计算”翻译成英文。",
    "中国的首都是哪里？只回答城市名。",
    "用一句话解释光合作用。",
    "判断正误：所有偶数都能被二整除。",
    "写出斐波那契数列的前六项。",
    "Python 中如何取得列表的最后一个元素？",
    "水在标准大气压下的沸点是多少摄氏度？",
    "将 3/4 转换成小数。",
    "用不超过十五个字说明什么是机器学习。",
    "如果今天是星期一，三天后是星期几？",
    "给出英文单词 reliability 的中文含义。",
    "一个正方形边长为 5，面积是多少？",
    "解释 CPU 与内存的基本关系，只用一句话。",
    "把数字 42 写成英文。",
    "判断正误：北京位于中国。",
    "Linux 中列出当前目录文件通常使用什么命令？",
    "一小时有多少秒？",
    "给“快速”写一个近义词。",
    "用一句话说明浮点量化的目的。",
]


PPL_TEXTS = [
    "海光加速器面向高性能计算和人工智能工作负载，软件栈需要同时保证正确性、稳定性与可维护性。",
    "模型量化通过降低权重和激活的表示位宽来减少存储与带宽开销，但必须评估由此产生的精度损失。",
    "混合专家模型只激活一部分专家参数，因此总参数量可以很大，而单个 token 的实际计算量相对有限。",
    "路由器会根据当前 token 的隐藏状态选择若干专家，微小的数值变化可能在候选分数接近时改变排序。",
    "可靠的回归测试应固定输入、随机种子、软件版本和模型权重，并分别记录算子误差与端到端质量。",
    "Matrix multiplication dominates many neural network workloads, while memory movement often determines practical inference speed.",
    "A numerical implementation is useful only when its accuracy, determinism, and performance are measured against a trusted baseline.",
    "Mixture of experts models route each token to a small subset of experts and combine their weighted outputs.",
    "Quantization reduces memory traffic by representing values with fewer bits, but the resulting approximation must remain bounded.",
    "Reproducible experiments separate model quality, quantization loss, and kernel implementation error instead of averaging them together.",
    "十七乘以二十三等于三百九十一。一个小时包含三千六百秒，四分之三等于零点七五。",
    "北京是中国的首都。水在标准大气压下通常在一百摄氏度沸腾，这些事实可用于基础质量探针。",
    "程序读取输入，执行确定的计算步骤，并把结果写入输出文件；退出码只能证明程序结束，不能证明结果正确。",
    "在同一输入上逐阶段比较中间张量，可以确定误差最早出现的位置，并避免把后续放大误认为根因。",
    "性能优化必须建立在正确性门禁之上，每次只改变一个变量，才能判断收益与回归分别来自哪里。",
    "Floating point formats divide bits among sign, exponent, and mantissa fields, creating different tradeoffs between range and precision.",
]


def serialize_logprobs(values):
    if values is None:
        return []
    return sorted(
        [
            {
                "token_id": int(token_id),
                "logprob": float(value.logprob),
                "rank": value.rank,
            }
            for token_id, value in values.items()
        ],
        key=lambda item: item["rank"] or 10**9,
    )


def main():
    model = os.environ["MXFP4_MODEL"]
    result_path = Path(os.environ["MXFP4_RESULT_PATH"])
    tensor_parallel_size = int(os.getenv("MXFP4_TP_SIZE", "2"))
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    chat_prompt_ids = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in CHAT_PROMPTS
    ]
    ppl_prompt_ids = [
        tokenizer(text, add_special_tokens=True).input_ids for text in PPL_TEXTS
    ]
    max_len = max(max(map(len, chat_prompt_ids)), max(map(len, ppl_prompt_ids))) + 16

    started = time.perf_counter()
    llm = LLM(
        model=model,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_len,
        max_num_batched_tokens=4096,
        max_num_seqs=32,
        gpu_memory_utilization=float(
            os.getenv("MXFP4_GPU_MEMORY_UTILIZATION", "0.82")
        ),
        enforce_eager=True,
        trust_remote_code=False,
        disable_log_stats=True,
        disable_custom_all_reduce=True,
    )
    load_seconds = time.perf_counter() - started

    chat_sampling = SamplingParams(
        temperature=0.0, max_tokens=8, logprobs=20
    )
    started = time.perf_counter()
    chat_outputs = llm.generate(
        [{"prompt_token_ids": ids} for ids in chat_prompt_ids],
        chat_sampling,
        use_tqdm=False,
    )
    chat_seconds = time.perf_counter() - started
    chat_cases = []
    for prompt, prompt_ids, request in zip(
        CHAT_PROMPTS, chat_prompt_ids, chat_outputs
    ):
        completion = request.outputs[0]
        chat_cases.append(
            {
                "prompt": prompt,
                "prompt_token_ids": prompt_ids,
                "generated_token_ids": list(completion.token_ids),
                "output": completion.text,
                "first_token_logprobs": serialize_logprobs(
                    completion.logprobs[0]
                ),
            }
        )

    ppl_sampling = SamplingParams(
        temperature=0.0, max_tokens=1, prompt_logprobs=1
    )
    started = time.perf_counter()
    ppl_outputs = llm.generate(
        [{"prompt_token_ids": ids} for ids in ppl_prompt_ids],
        ppl_sampling,
        use_tqdm=False,
    )
    ppl_seconds = time.perf_counter() - started
    total_nll = 0.0
    total_tokens = 0
    ppl_cases = []
    for text, token_ids, request in zip(PPL_TEXTS, ppl_prompt_ids, ppl_outputs):
        case_nll = 0.0
        case_tokens = 0
        for token_id, values in zip(token_ids[1:], request.prompt_logprobs[1:]):
            if values is None or token_id not in values:
                raise RuntimeError(f"chosen token {token_id} missing from prompt logprobs")
            case_nll -= float(values[token_id].logprob)
            case_tokens += 1
        total_nll += case_nll
        total_tokens += case_tokens
        ppl_cases.append(
            {
                "text": text,
                "token_count": case_tokens,
                "nll": case_nll,
                "mean_nll": case_nll / case_tokens,
            }
        )

    payload = {
        "mode": MODE,
        "model": model,
        "tensor_parallel_size": tensor_parallel_size,
        "load_seconds": load_seconds,
        "chat_seconds": chat_seconds,
        "ppl_seconds": ppl_seconds,
        "chat_cases": chat_cases,
        "ppl": {
            "token_count": total_tokens,
            "nll": total_nll,
            "mean_nll": total_nll / total_tokens,
            "perplexity": math.exp(total_nll / total_tokens),
            "cases": ppl_cases,
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "mode": MODE,
                "ppl": payload["ppl"]["perplexity"],
                "ppl_tokens": total_tokens,
                "chat_outputs": [case["output"] for case in chat_cases],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
