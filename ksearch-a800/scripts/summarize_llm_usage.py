#!/usr/bin/env python3
"""汇总一个或多个 LLM usage.jsonl 文件的 token 用量。

用法：
    python scripts/summarize_llm_usage.py baseline/ksearch/rmsnorm_h4096/run_seed0/usage.jsonl
    python scripts/summarize_llm_usage.py baseline/ksearch --output baseline/usage/ksearch_summary.json

输入可为 usage.jsonl 文件或包含该文件的目录。输出不含 API key、prompt 或 response。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def usage_files(inputs: list[Path]) -> list[Path]:
    files = []
    for path in inputs:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("usage.jsonl")))
        else:
            raise FileNotFoundError(path)
    return sorted(set(files))


def as_int(value) -> int:
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path, help="usage.jsonl 文件或其上层目录")
    parser.add_argument("--output", type=Path, help="可选：写入 JSON 汇总")
    args = parser.parse_args()

    totals = Counter()
    by_model: dict[str, Counter] = {}
    files = usage_files(args.paths)
    for path in files:
        for raw in path.read_text().splitlines():
            if not raw.strip():
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                totals["invalid_records"] += 1
                continue
            model = str(rec.get("model") or "unknown")
            target = by_model.setdefault(model, Counter())
            input_tokens = as_int(rec.get("input_tokens", rec.get("prompt_tokens")))
            cached_raw = rec.get("input_cached_tokens")
            cache_usage_reported = rec.get("cache_usage_reported")
            if cache_usage_reported is None:
                # Backward compatibility: normalized records with a numeric
                # cached count reported it; legacy records with no count did not.
                cache_usage_reported = cached_raw is not None
            cache_usage_reported = bool(cache_usage_reported)
            cached_tokens = as_int(cached_raw) if cache_usage_reported else 0
            output_tokens = as_int(rec.get("output_tokens", rec.get("completion_tokens")))
            # reasoning_tokens 是 output_tokens 的子集（provider 单列时才有值）。
            reasoning_raw = rec.get("reasoning_tokens")
            reasoning_usage_reported = rec.get("reasoning_usage_reported")
            if reasoning_usage_reported is None:
                reasoning_usage_reported = reasoning_raw is not None
            reasoning_tokens = as_int(reasoning_raw) if reasoning_usage_reported else 0
            for counter in (totals, target):
                counter["calls"] += 1
                counter["input_tokens"] += input_tokens
                counter["input_cached_tokens"] += cached_tokens
                if cache_usage_reported:
                    counter["cache_usage_reported_calls"] += 1
                    counter["input_uncached_tokens"] += max(0, input_tokens - cached_tokens)
                else:
                    counter["cache_usage_unreported_calls"] += 1
                    counter["input_tokens_cache_status_unknown"] += input_tokens
                counter["output_tokens"] += output_tokens
                if reasoning_usage_reported:
                    counter["reasoning_usage_reported_calls"] += 1
                    counter["reasoning_tokens"] += reasoning_tokens
                else:
                    counter["reasoning_usage_unreported_calls"] += 1

    payload = {
        "schema": "llm-usage-summary-v1",
        "usage_files": [str(path) for path in files],
        "totals": dict(totals),
        "by_model": {model: dict(counter) for model, counter in sorted(by_model.items())},
    }
    payload["totals"]["total_tokens"] = (
        payload["totals"].get("input_tokens", 0) + payload["totals"].get("output_tokens", 0)
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
