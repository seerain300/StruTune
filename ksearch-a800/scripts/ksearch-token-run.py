#!/usr/bin/env python3
"""K-Search 运行包装器：给 OpenAI SDK 客户端打补丁，记录每次 LLM 调用的 token usage。

不改 K-Search 源码；在 generate_kernels_and_eval.py 之前 import 本模块生效。
用法：
    KSEARCH_USAGE_LOG=<run_dir>/usage.jsonl python ksearch-token-run.py <原样透传给 generate_kernels_and_eval.py 的参数>

JSONL 字段遵循本项目的 usage-v1 schema。不会记录 prompt、response、API key 或请求头；
只记录输入 token、可从 provider response 获得的缓存输入 token、输出 token、reasoning token
（provider 单列时；它是 output_tokens 的子集，不是额外增量）和模型名。
"""

import json
import os
import runpy
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

USAGE_LOG = Path(os.environ.get("KSEARCH_USAGE_LOG", "ksearch_usage.jsonl"))

import openai  # noqa: E402

_orig_init = openai.OpenAI.__init__
_patch_installed = False


def _field(obj, *names):
    """从 SDK 对象或 dict 安全读取第一个存在的字段。"""
    if obj is None:
        return None
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _as_int(value):
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _usage_record(response):
    """Normalize OpenAI-compatible usage variants to one stable, non-sensitive record."""
    usage = _field(response, "usage")
    input_tokens = _as_int(_field(usage, "prompt_tokens", "input_tokens"))
    output_tokens = _as_int(_field(usage, "completion_tokens", "output_tokens"))
    total_tokens = _as_int(_field(usage, "total_tokens"))

    # Chat Completions normally exposes prompt_tokens_details; Responses-style
    # providers commonly use input_tokens_details.  Some compatible providers
    # expose cache_read_input_tokens instead.
    input_details = _field(usage, "prompt_tokens_details", "input_tokens_details")
    cached_input_tokens = _as_int(
        _field(input_details, "cached_tokens", "cache_read_input_tokens")
    )
    if cached_input_tokens is None:
        cached_input_tokens = _as_int(_field(usage, "cached_tokens", "cache_read_input_tokens"))
    cache_usage_reported = cached_input_tokens is not None
    uncached_input_tokens = (
        max(0, input_tokens - cached_input_tokens)
        if input_tokens is not None and cache_usage_reported
        else None
    )

    # Reasoning/thinking tokens when the provider itemizes them.  NOTE: these are
    # a SUBSET of output_tokens (completion_tokens already includes reasoning);
    # do not add them on top of output_tokens when totaling.
    output_details = _field(usage, "completion_tokens_details", "output_tokens_details")
    reasoning_tokens = _as_int(_field(output_details, "reasoning_tokens"))
    reasoning_usage_reported = reasoning_tokens is not None

    return {
        "schema": "llm-usage-v1",
        "event": "completion",
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "method": os.environ.get("LLM_USAGE_METHOD", "ksearch"),
        "run_id": os.environ.get("LLM_USAGE_RUN_ID"),
        "model": _field(response, "model"),
        "input_tokens": input_tokens,
        "input_cached_tokens": cached_input_tokens,
        "input_uncached_tokens": uncached_input_tokens,
        "cache_usage_reported": cache_usage_reported,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "reasoning_usage_reported": reasoning_usage_reported,
        "total_tokens": total_tokens,
        # Backward-compatible aliases consumed by existing tooling.
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
    }


def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)

    def _log_usage(resp, prompt_text=None):
        try:
            rec = _usage_record(resp)
            # Opt-in prompt fingerprint (sha256, no content) for cache-hit
            # forensics: lets a hit be paired with its byte-identical
            # predecessor (or expose SDK-retry pairs where the failed first
            # attempt was never logged).  Enable with KSEARCH_LOG_PROMPT_HASH=1.
            if prompt_text is not None and os.environ.get("KSEARCH_LOG_PROMPT_HASH") == "1":
                import hashlib

                rec["prompt_sha256"] = hashlib.sha256(
                    json.dumps(prompt_text, ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest()
            with open(USAGE_LOG, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass
        return resp

    _orig_create = self.chat.completions.create

    def _create(*a, **kw):
        prompt_probe = None
        if os.environ.get("KSEARCH_LOG_PROMPT_HASH") == "1":
            try:
                msgs = kw.get("messages") or (a[0] if a else None)
                prompt_probe = msgs
            except Exception:
                prompt_probe = None
        # 网关 5xx/限速/连接错误的显式退避重试（SDK 默认 2 次对 502 风暴不够，
        # 012 题曾因连续 502 死亡）。指数退避 2→5→15→30→60s，最多 6 次尝试。
        delays = [0, 2, 5, 15, 30, 60]
        last_err = None
        for attempt, delay in enumerate(delays):
            if delay:
                time.sleep(delay)
            try:
                resp = _orig_create(*a, **kw)
                return _log_usage(resp, prompt_text=prompt_probe)
            except Exception as e:
                last_err = e
                msg = str(e)
                retryable = any(
                    t in msg
                    for t in ("502", "503", "504", "Bad Gateway", "429",
                              "rate limit", "Rate limit", "timeout", "Timeout",
                              "connection", "Connection", "reset by peer")
                )
                if not retryable or attempt == len(delays) - 1:
                    raise
        raise last_err

    self.chat.completions.create = _create

    # K-Search routes gpt-5*/o3* model names to client.responses.create;
    # cover that path too so usage accounting does not silently miss calls.
    responses = getattr(self, "responses", None)
    if responses is not None and hasattr(responses, "create"):
        _orig_responses_create = responses.create

        def _responses_create(*a, **kw):
            return _log_usage(_orig_responses_create(*a, **kw))

        responses.create = _responses_create


if not _patch_installed:
    openai.OpenAI.__init__ = _patched_init
    _patch_installed = True

# 以原入口执行（cwd 应在 K-Search 仓库内，由 ksearch-run.sh 保证）
sys.argv = ["generate_kernels_and_eval.py"] + sys.argv[1:]
runpy.run_path("generate_kernels_and_eval.py", run_name="__main__")
