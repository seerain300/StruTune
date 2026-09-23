"""SOL-ExecBench task backend for K-Search.

Loads a benchmark problem (definition.json + workload.jsonl, e.g. under
``data/benchmark/L1/<name>``) from the official SOL-ExecBench dataset and
evaluates candidates through the official ``sol-execbench`` CLI running in the
SOL virtualenv.  Custom input generation, per-workload tolerances, and
anti-reward-hack checks therefore match the official evaluation exactly.

Optimization feedback samples a fixed (seeded) subset of workloads; the final
evaluation covers all workloads through the same pipeline.

This module must stay importable from the K-Search generation environment
(mtmc env); it never imports sol_execbench directly, it only shells out to the
official CLI.
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from k_search.tasks.task_base import (
    BuildSpec,
    EvalResult,
    Solution as TaskSolution,
    SourceFile,
    SupportedLanguages,
    load_ksearch_solution_json,
    solution_from_json_dict,
)

DEFAULT_SOL_ROOT = Path("/data1/workspace/ziming/dataset/SOL-ExecBench")

# Cross-run cache for reference latencies (search-feedback signal only; the
# final evaluation always re-times the reference through the official CLI).
SOL_REF_LATENCY_CACHE_ROOT = Path(
    "/data1/workspace/weihongren/baseline/ksearch-sol-execbench/.ref_latency_cache"
)

_KSEARCH_LANGUAGE_TO_SOL = {
    "triton": "triton",
    "python": "pytorch",
    "pytorch": "pytorch",
    "cuda": "cuda_cpp",
    "cpp": "cuda_cpp",
    "cuda_cpp": "cuda_cpp",
}


@dataclass
class SolExecBenchEvalConfig:
    """Parameters forwarded to the official SOL BenchmarkConfig + CLI."""

    warmup_runs: int = 3
    iterations: int = 10
    # Needed for speedup feedback (reference latency is measured in the same
    # staged process, per official semantics).
    benchmark_reference: bool = True
    correctness_only: bool = False
    # Official input-generation seed; fixed across methods and search seeds.
    eval_seed: int = 200
    # CLI --timeout budget for the whole evaluation subprocess.  With unified
    # timing (iterations=100), a slow reference can legitimately take ~10 min
    # per workload on the first (cache-filling) round, so budgets are generous.
    timeout_seconds: int = 1200
    # Extra per-workload seconds added to timeout_seconds when many workloads
    # are evaluated in one subprocess.
    per_workload_timeout_seconds: int = 1800


def _sol_python_and_cli(sol_root: Path) -> tuple[Path, Path]:
    sol_python = sol_root / ".venv/bin/python"
    sol_cli = sol_root / ".venv/bin/sol-execbench"
    if not sol_cli.is_file():
        raise FileNotFoundError(
            f"sol-execbench CLI not found at {sol_cli}; expected the official "
            "SOL-ExecBench virtualenv under the dataset root."
        )
    return sol_python, sol_cli


def _normalize_definition(definition: dict) -> tuple[dict, list[str]]:
    """Defensive, in-memory normalization; the on-disk dataset stays untouched."""
    normalized: list[str] = []
    if definition.get("hf_id") == "":
        definition.pop("hf_id", None)
        normalized.append("removed empty optional definition.hf_id")
    return definition, normalized


def _signature_warning(solution: TaskSolution, definition: dict) -> Optional[str]:
    """Best-effort entry-signature check; returns a warning string or None."""
    try:
        import ast

        entry_file, entry_name = solution.spec.entry_point.split("::", 1)
        source = next(
            (s for s in (solution.sources or []) if s.path == entry_file), None
        )
        if source is None:
            return f"entry source {entry_file!r} is missing"
        tree = ast.parse(source.content, filename=entry_file)
        fn = next(
            (
                node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == entry_name
            ),
            None,
        )
        if fn is None:
            return f"entry function {entry_name!r} is missing from {entry_file}"
        if fn.args.vararg or fn.args.kwarg:
            return None
        actual = [a.arg for a in fn.args.posonlyargs + fn.args.args]
        expected = list(definition.get("inputs", {}))
        if solution.spec.destination_passing_style:
            expected += list(definition.get("outputs", {}))
        if actual != expected:
            return f"entry signature mismatch: expected {expected}, got {actual}"
    except SyntaxError as e:
        return f"entry source failed to parse: {e}"
    except Exception:
        return None
    return None


class SolExecBenchTask:
    """K-Search task backed by the official SOL-ExecBench CLI."""

    def __init__(
        self,
        *,
        problem_dir: Path,
        sol_root: Path = DEFAULT_SOL_ROOT,
        warmup_runs: int = 3,
        iterations: int = 10,
        benchmark_reference: bool = True,
        correctness_only: bool = False,
        eval_seed: int = 200,
        timeout_seconds: int = 1200,
        per_workload_timeout_seconds: int = 1800,
        num_feedback_workloads: int = 5,
        feedback_workloads: Optional[List[str]] = None,
        artifacts_dir: Optional[str] = None,
        seed: int = 0,
    ) -> None:
        self.sol_root = Path(sol_root)
        self.problem_dir = Path(problem_dir)
        if not self.problem_dir.is_dir():
            raise FileNotFoundError(f"SOL problem dir not found: {self.problem_dir}")

        definition_path = self.problem_dir / "definition.json"
        workload_path = self.problem_dir / "workload.jsonl"
        raw_definition = json.loads(definition_path.read_text(encoding="utf-8"))
        self._definition, self._definition_normalizations = _normalize_definition(
            raw_definition
        )
        self._workload_records: List[dict] = [
            json.loads(line)
            for line in workload_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self._workload_records:
            raise ValueError(f"workload file is empty: {workload_path}")

        self._eval_config = SolExecBenchEvalConfig(
            warmup_runs=int(warmup_runs),
            iterations=int(iterations),
            benchmark_reference=bool(benchmark_reference),
            correctness_only=bool(correctness_only),
            eval_seed=int(eval_seed),
            timeout_seconds=int(timeout_seconds),
            per_workload_timeout_seconds=int(per_workload_timeout_seconds),
        )
        self._rng = random.Random(int(seed))
        self._seed = int(seed)
        self._init_num_feedback_workloads = int(num_feedback_workloads)
        self._init_feedback_workloads = list(feedback_workloads) if feedback_workloads else None
        self._selected_workloads: List[dict] = []
        self._artifacts_dir = artifacts_dir

        # Last-round feedback state consumed by generators.
        self._last_round_passed_count: int = 0
        self._last_round_total_workloads: int = 0
        self._last_round_trace_logs_for_prompt: str = ""
        self._last_round_traces: List[dict] = []
        self._last_round_summary_line: str = ""
        # Reference latency cache per workload uuid: the official driver re-times
        # the reference on every call when benchmark_reference=True, which is
        # needlessly expensive for fixed feedback workloads across search rounds.
        # Cached values only affect in-loop feedback; the final evaluation always
        # re-times the reference through the official pipeline.
        self._ref_latency_by_uuid: Dict[str, float] = {}
        self._load_ref_latency_cache_from_disk()

        _sol_python_and_cli(self.sol_root)  # fail fast when the venv is missing
        self._select_feedback_workloads()

    # -------- Construction helpers --------

    @staticmethod
    def resolve_problem_dir(sol_root: Path, definition_name: str) -> Path:
        """Accept 'L1/094_...' or a bare name; search benchmark subsets."""
        name = str(definition_name or "").strip().strip("/")
        candidates = [Path(sol_root) / "data" / "benchmark" / name]
        if "/" in name:
            candidates.insert(0, Path(sol_root) / "data" / "benchmark" / name)
        for cand in candidates:
            if (cand / "definition.json").is_file():
                return cand
        # Last resort: scan benchmark subsets for a matching directory name.
        base = Path(sol_root) / "data" / "benchmark"
        if base.is_dir():
            tail = name.split("/")[-1]
            for subset in sorted(base.iterdir()):
                if not subset.is_dir():
                    continue
                cand = subset / tail
                if (cand / "definition.json").is_file():
                    return cand
        raise FileNotFoundError(
            f"SOL problem {definition_name!r} not found under {base}"
        )

    @classmethod
    def from_cli_args(
        cls,
        *,
        sol_root: str,
        definition_name: str,
        warmup_runs: int,
        iterations: int,
        feedback_workloads: Optional[List[str]],
        num_feedback_workloads: int,
        artifacts_dir: Optional[str],
        seed: int,
    ) -> "SolExecBenchTask":
        root = Path(sol_root)
        problem_dir = cls.resolve_problem_dir(root, definition_name)
        return cls(
            problem_dir=problem_dir,
            sol_root=root,
            warmup_runs=warmup_runs,
            iterations=iterations,
            feedback_workloads=feedback_workloads,
            num_feedback_workloads=num_feedback_workloads,
            artifacts_dir=artifacts_dir,
            seed=seed,
        )

    # -------- Task protocol --------

    @property
    def name(self) -> str:
        return str(self._definition.get("name") or self.problem_dir.name)

    def get_config_for_logging(self) -> Dict[str, Any]:
        return {
            "task_backend": "sol_execbench",
            "definition": self.name,
            "problem_dir": str(self.problem_dir),
            "sol_root": str(self.sol_root),
            "workload_count": len(self._workload_records),
            "num_feedback_workloads": self._init_num_feedback_workloads,
            "feedback_workloads": (
                [str(w.get("uuid")) for w in self._selected_workloads]
                if self._selected_workloads
                else None
            ),
            "search_seed": self._seed,
            "eval_seed": self._eval_config.eval_seed,
            "warmup_runs": self._eval_config.warmup_runs,
            "iterations": self._eval_config.iterations,
            "benchmark_reference": self._eval_config.benchmark_reference,
            "correctness_only": self._eval_config.correctness_only,
            "definition_normalizations": self._definition_normalizations,
        }

    def get_definition_text(self, language: str | None = None) -> str:
        definition = self._definition
        axes = definition.get("axes") or {}
        axes_str = "\nAxes:\n"
        for axis_name, axis in axes.items():
            axis_type = str((axis or {}).get("type") or "var")
            if axis_type == "const":
                axes_str += f"  {axis_name}: constant = {axis.get('value')}"
            else:
                axes_str += f"  {axis_name}: variable"
            if axis.get("description"):
                axes_str += f" ({axis['description']})"
            axes_str += "\n"

        def _spec_str(spec: dict) -> str:
            shape = spec.get("shape")
            shape_str = "scalar" if shape is None else f"[{', '.join(map(str, shape))}]"
            out = f"{shape_str} ({spec.get('dtype')})"
            if spec.get("description"):
                out += f" - {spec['description']}"
            return out

        inputs_str = "\nInputs:\n"
        for in_name, spec in (definition.get("inputs") or {}).items():
            inputs_str += f"  {in_name}: {_spec_str(spec)}\n"
        outputs_str = "\nOutputs:\n"
        for out_name, spec in (definition.get("outputs") or {}).items():
            outputs_str += f"  {out_name}: {_spec_str(spec)}\n"

        tolerance = self._workload_records[0].get("tolerance") or {}
        tol_str = ""
        if tolerance:
            tol_str = (
                f"\nNumerical tolerance (typical workload): "
                f"max_atol={tolerance.get('max_atol')}, max_rtol={tolerance.get('max_rtol')}\n"
            )

        custom_str = ""
        if definition.get("custom_inputs_entrypoint"):
            custom_str = (
                "\nNote: inputs are generated by the benchmark's own custom input "
                f"generator ({definition['custom_inputs_entrypoint']}); tensors are "
                "already on GPU when run() is called.\n"
            )

        return str(
            f"Name: {definition.get('name')}\n"
            f"Description: {definition.get('description')}\n"
            f"{axes_str}{inputs_str}{outputs_str}{tol_str}{custom_str}\n"
            f"Reference Implementation:\n{definition.get('reference')}"
        ).strip()

    def get_solution(self, solution_name: str) -> Optional[TaskSolution]:
        """Load a persisted K-Search solution JSON by name or path."""
        try:
            d = load_ksearch_solution_json(
                solution_ref=solution_name,
                definition_name=self.name,
                artifacts_dir=self._artifacts_dir,
            )
            return solution_from_json_dict(d)
        except Exception as e:
            print(f"[sol-task] failed to load solution {solution_name!r}: {e}")
            return None

    def code_for_world_model_from_raw(self, *, raw: Any, language: str) -> str:
        return str(raw or "")

    def seed_eval_for_base_solution(
        self, *, base_solution: TaskSolution, config: Any = None
    ) -> EvalResult:
        return self.run_benchmark(solution=base_solution, config=config, round_num=None)

    # -------- Prompt hooks --------

    def get_per_task_requirement_text(
        self, *, language: str, target_gpu: str, phase: str = ""
    ) -> str:
        try:
            from k_search.tasks.flashinfer_bench.prompts import (
                per_task_requirement_text,
            )

            base = str(
                per_task_requirement_text(
                    language=str(language),
                    target_gpu=str(target_gpu),
                    phase=str(phase or ""),
                )
                or ""
            ).strip()
        except Exception:
            base = ""

        # 强化版禁止调库（KSEARCH_STRICT_NO_LIB=1 时替换默认守卫）
        if os.environ.get("KSEARCH_STRICT_NO_LIB") == "1":
            sol_specific = (
                "CRITICAL: Your run() function MUST NOT call any torch library computation functions:\n"
                "- BANNED: torch.matmul, torch.mm, torch.bmm, torch.addmm, torch.einsum\n"
                "- BANNED: F.linear, nn.functional.linear\n"
                "- BANNED: F.conv1d, F.conv2d, F.conv3d, F.conv_transpose1d, F.conv_transpose2d\n"
                "- BANNED: torch.fft.rfft, torch.fft.fft, torch.fft.irfft, torch.fft.ifft\n"
                "- BANNED: torch.cumsum, torch.sort, torch.topk, torch.unique\n\n"
                "ALL computation must be performed by Triton kernels (@triton.jit). "
                "The ONLY torch operations allowed are tensor creation (torch.zeros/empty/empty_like), "
                "shape manipulation (.view/.reshape/.transpose/.contiguous/.permute), "
                "and data movement (.to()).\n\n"
                "If you need matrix multiply → use tl.dot() in your kernel.\n"
                "If you need convolution → implement sliding window + accumulate in your kernel.\n"
                "If you need FFT → implement butterfly operations in your kernel.\n"
                "If you need reduction → use tl.sum()/tl.max() in your kernel.\n\n"
                "A solution that calls any banned function will be REJECTED.\n\n"
                "SOL-ExecBench interface requirements:\n"
                "- The `run(...)` entry function's parameter names and order must EXACTLY "
                "match the Inputs listed in the definition.\n"
                "- run() must return the outputs in the same order as the definition's "
                "Outputs list (single tensor or tuple of tensors; do not mutate inputs).\n"
                "- Numerical correctness is checked per workload against the reference with "
                "the tolerance shown in the definition; match reference dtype/shape exactly.\n"
                "- The benchmark measures wall latency on already-on-GPU tensors; do not "
                "add synchronization or CPU-GPU transfers inside run().\n"
            )
        else:
            sol_specific = (
                # 本地新增（L2 批次起生效）：与上游 KernelBench 后端的 NO_TORCH_FALLBACK_WARNING
                # 同义的防退化守卫——上游未把它接入 FlashInfer/SOL 路径（gemm 曾因此退化为
                # 调用 reference 同款 torch.matmul）。L1 10 题为无守卫口径，已在 manifest 标注。
                "**IMPORTANT**: Avoid using torch functions as fallbacks in your implementation. "
                "Do not use try/catch blocks that fall back to torch operations. Your custom kernel "
                "should be the primary implementation path and handle all cases directly.\n\n"
                "SOL-ExecBench interface requirements:\n"
                "- The `run(...)` entry function's parameter names and order must EXACTLY "
                "match the Inputs listed in the definition (same identifiers, no renaming, "
                "no *args/**kwargs, no default values).\n"
                "- run() must return the outputs in the same order as the definition's "
                "Outputs list (single tensor or tuple of tensors; do not mutate inputs).\n"
                "- Numerical correctness is checked per workload against the reference with "
                "the tolerance shown in the definition; match reference dtype/shape exactly.\n"
                "- The benchmark measures wall latency on already-on-GPU tensors; do not "
                "add synchronization, allocations of the output buffer per iteration beyond "
                "what is needed, or CPU-GPU transfers inside run().\n"
            )
        return "\n".join(part for part in (sol_specific, base) if part).strip()

    def get_baseline_targets_text(self) -> str:
        ref_latencies = [
            float(t["evaluation"]["performance"]["reference_latency_ms"])
            for t in self._last_round_traces
            if (t.get("evaluation") or {}).get("performance", {}).get("reference_latency_ms")
        ]
        if not ref_latencies:
            return "Baseline: official PyTorch reference; target speedup_vs_ref > 1.0x."
        mean_ref = sum(ref_latencies) / len(ref_latencies)
        return (
            f"Baseline: official PyTorch reference (mean latency {mean_ref:.4f} ms over "
            f"{len(ref_latencies)} feedback workload(s)); target speedup_vs_ref > 1.0x."
        )

    def has_last_round_feedback_trace(self) -> bool:
        return bool(self._last_round_trace_logs_for_prompt) or bool(
            self._last_round_summary_line
        )

    def get_last_round_trace_logs_for_prompt(self) -> str:
        return self._last_round_trace_logs_for_prompt

    def get_last_round_passed_count(self) -> int:
        return self._last_round_passed_count

    def get_last_round_total_workloads(self) -> int:
        return self._last_round_total_workloads

    # -------- Workload selection --------

    def _select_feedback_workloads(self) -> None:
        records = list(self._workload_records)
        explicit = self._init_feedback_workloads
        if explicit:
            by_uuid = {str(r.get("uuid")): r for r in records}
            selected = [by_uuid[u] for u in explicit if u in by_uuid]
            missing = [u for u in explicit if u not in by_uuid]
            if missing:
                print(f"[sol-task] WARNING: unknown feedback workload uuids ignored: {missing}")
        else:
            n = min(int(self._init_num_feedback_workloads), len(records))
            selected = self._rng.sample(records, n)
        if not selected:
            raise ValueError("no feedback workloads selected")
        self._selected_workloads = selected

    def set_selected_workloads(self, selected_uuids: List[str]) -> None:
        by_uuid = {str(r.get("uuid")): r for r in self._workload_records}
        selected = [by_uuid[u] for u in selected_uuids if u in by_uuid]
        if selected:
            self._selected_workloads = selected

    # -------- Solution conversion --------

    def to_sol_solution(self, solution: TaskSolution) -> dict:
        spec = solution.spec
        lang_value = (
            spec.language.value
            if isinstance(spec.language, SupportedLanguages)
            else str(spec.language or "")
        )
        sol_language = _KSEARCH_LANGUAGE_TO_SOL.get(lang_value.strip().lower())
        if sol_language is None:
            raise ValueError(
                f"unsupported K-Search language for SOL-ExecBench: {lang_value!r}"
            )
        deps = list(spec.dependencies or [])
        if sol_language == "triton":
            deps = list(dict.fromkeys([*deps, "torch", "triton"]))
        elif sol_language == "pytorch":
            deps = list(dict.fromkeys([*deps, "torch"]))
        warning = _signature_warning(solution, self._definition)
        if warning:
            # Non-fatal: the CLI will surface the concrete error to the
            # optimization loop, which can self-correct next round.
            print(f"[sol-task] WARNING: {warning}")
        return {
            "name": f"{solution.name}_sol_execbench",
            "definition": self.name,
            "author": str(solution.author or "ksearch"),
            "description": (solution.description or "") + " (K-Search candidate)",
            "spec": {
                "languages": [sol_language],
                "target_hardware": ["LOCAL"],
                "entry_point": str(spec.entry_point or "main.py::run"),
                "dependencies": deps,
                "destination_passing_style": bool(spec.destination_passing_style),
                "binding": None,
            },
            "sources": [
                {"path": s.path, "content": s.content} for s in (solution.sources or [])
            ],
        }

    # -------- Benchmark execution --------

    def _artifacts_eval_dir(self) -> Optional[Path]:
        if not self._artifacts_dir:
            return None
        try:
            from k_search.utils.paths import get_ksearch_artifacts_dir

            root = get_ksearch_artifacts_dir(
                base_dir=self._artifacts_dir, task_name=self.name
            )
            out = Path(root) / "eval" / self.name
            out.mkdir(parents=True, exist_ok=True)
            return out
        except Exception:
            return None

    @staticmethod
    def _spawn_sol_cli(cmd: list, env: dict, cli_timeout: int):
        """运行 SOL CLI 子进程；超时返回错误字符串，否则返回 CompletedProcess。"""
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=cli_timeout + 300,
                env=env,
            )
        except subprocess.TimeoutExpired as e:
            tail = str(e.stderr or e.stdout or "")[-2000:]
            return f"SOL CLI subprocess timed out after {cli_timeout + 300}s\n{tail}"

    def _run_sol_cli(
        self,
        *,
        sol_solution: dict,
        workloads: List[dict],
        config: SolExecBenchEvalConfig,
    ) -> tuple[List[dict], str]:
        """Stage files and invoke the official CLI; returns (traces, error_text)."""
        _, sol_cli = _sol_python_and_cli(self.sol_root)

        with tempfile.TemporaryDirectory(prefix="ksearch_sol_") as tmp:
            tmp_root = Path(tmp)
            def_file = tmp_root / "definition.json"
            wl_file = tmp_root / "workload.jsonl"
            sol_file = tmp_root / "solution.json"
            cfg_file = tmp_root / "config.json"
            def_file.write_text(
                json.dumps(self._definition, indent=2) + "\n", encoding="utf-8"
            )
            wl_file.write_text(
                "".join(json.dumps(w) + "\n" for w in workloads), encoding="utf-8"
            )
            sol_file.write_text(json.dumps(sol_solution, indent=2), encoding="utf-8")
            cfg_file.write_text(
                json.dumps(
                    {
                        "warmup_runs": config.warmup_runs,
                        "iterations": config.iterations,
                        "benchmark_reference": config.benchmark_reference,
                        "correctness_only": config.correctness_only,
                        "seed": config.eval_seed,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            cli_timeout = max(
                config.timeout_seconds,
                config.per_workload_timeout_seconds * len(workloads),
            )
            cmd = [
                str(sol_cli),
                "--definition", str(def_file),
                "--workload", str(wl_file),
                "--solution", str(sol_file),
                "--config", str(cfg_file),
                "--timeout", str(cli_timeout),
                "--json",
            ]
            env = dict(os.environ)
            env["PATH"] = f"{str(sol_cli.parent)}:{env.get('PATH', '')}"
            env["FLASHINFER_TRACE_DIR"] = str(self.sol_root.resolve())

            # GPU 池模式（KSEARCH_GPU_POOL="5,6"）：benchmark 子进程动态从池中领卡，
            # LLM 生成阶段不占卡；未设置时沿用调用方固定的 CUDA_VISIBLE_DEVICES。
            pool_spec = os.environ.get("KSEARCH_GPU_POOL", "").strip()
            if pool_spec:
                from k_search.utils.gpu_pool import gpu_slot

                with gpu_slot(pool_spec) as gpu_id:
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    result = self._spawn_sol_cli(cmd, env, cli_timeout)
            else:
                result = self._spawn_sol_cli(cmd, env, cli_timeout)
            if isinstance(result, str):  # 超时错误串
                return [], result

            traces: List[dict] = []
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and (obj.get("workload") or {}).get("uuid"):
                    traces.append(obj)

            error_parts: List[str] = []
            if result.returncode != 0:
                error_parts.append(f"sol-execbench exit code {result.returncode}")
            if not traces:
                error_parts.append("no trace JSON produced")
            if error_parts and result.stderr.strip():
                error_parts.append("stderr tail:\n" + result.stderr.strip()[-2000:])
            return traces, "\n".join(error_parts)

    # -------- Reference-latency disk cache (search feedback only) --------

    def _ref_latency_cache_path(self) -> Path:
        cfg = self._eval_config
        return (
            SOL_REF_LATENCY_CACHE_ROOT
            / f"{self.name}_w{int(cfg.warmup_runs)}_i{int(cfg.iterations)}_s{int(cfg.eval_seed)}.json"
        )

    def _load_ref_latency_cache_from_disk(self) -> None:
        path = self._ref_latency_cache_path()
        if not path.is_file():
            return
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if str(obj.get("definition") or "") != self.name:
                return
            per_wl = obj.get("per_workload") or {}
            for k, v in per_wl.items():
                if isinstance(v, (int, float)) and float(v) > 0:
                    self._ref_latency_by_uuid[str(k)] = float(v)
        except Exception:
            pass

    def _save_ref_latency_cache_to_disk(self) -> None:
        if not self._ref_latency_by_uuid:
            return
        try:
            path = self._ref_latency_cache_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema": "ksearch-sol-ref-latency-cache-v1",
                "definition": self.name,
                "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "note": "search-feedback signal only; final evaluation re-times the reference",
                "per_workload": dict(self._ref_latency_by_uuid),
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[sol-task] WARNING: failed to save ref-latency cache: {e}")

    def _feedback_logs_from_traces(
        self, traces: List[dict], workloads: List[dict]
    ) -> str:
        """Human/prompt-readable per-workload feedback (mirrors flashinfer task style)."""
        lines: List[str] = []
        for wl in workloads:
            uuid = str(wl.get("uuid"))
            trace = next(
                (t for t in traces if str((t.get("workload") or {}).get("uuid")) == uuid),
                None,
            )
            axes = ", ".join(f"{k}={v}" for k, v in sorted((wl.get("axes") or {}).items()))
            if trace is None:
                lines.append(f"- workload {uuid} (axes: {axes}): NO_TRACE")
                continue
            evaluation = trace.get("evaluation") or {}
            status = evaluation.get("status", "UNKNOWN")
            correctness = evaluation.get("correctness") or {}
            performance = evaluation.get("performance") or {}
            if status == "PASSED":
                lines.append(
                    f"- workload {uuid} (axes: {axes}): PASSED, "
                    f"latency {performance.get('latency_ms')} ms, "
                    f"reference {performance.get('reference_latency_ms')} ms, "
                    f"speedup {performance.get('speedup_factor')}x"
                )
            else:
                detail = f"- workload {uuid} (axes: {axes}): {status}"
                if correctness:
                    detail += (
                        f", max_abs_err={correctness.get('max_absolute_error')}, "
                        f"max_rel_err={correctness.get('max_relative_error')}"
                    )
                lines.append(detail)
                log = str(evaluation.get("log") or "").strip()
                if log:
                    lines.append("  log excerpt: " + log[-1200:])
        return "\n".join(lines).strip()

    def _eval_result_from_traces(
        self,
        *,
        traces: List[dict],
        workloads: List[dict],
        error_text: str,
    ) -> EvalResult:
        total = len(workloads)
        passed = 0
        speedups: List[float] = []
        latencies: List[float] = []
        ref_latencies: List[float] = []
        first_failure_status: Optional[str] = None
        log_excerpts: List[str] = []

        traces_by_uuid = {
            str((t.get("workload") or {}).get("uuid")): t for t in traces
        }
        for wl in workloads:
            uuid = str(wl.get("uuid"))
            trace = traces_by_uuid.get(uuid)
            if trace is None:
                if first_failure_status is None:
                    first_failure_status = "NO_TRACE"
                continue
            evaluation = trace.get("evaluation") or {}
            status = str(evaluation.get("status") or "UNKNOWN")
            if status == "PASSED":
                passed += 1
                performance = evaluation.get("performance") or {}
                for key, bucket in (
                    ("speedup_factor", speedups),
                    ("latency_ms", latencies),
                    ("reference_latency_ms", ref_latencies),
                ):
                    value = performance.get(key)
                    if isinstance(value, (int, float)) and float(value) > 0:
                        bucket.append(float(value))
            else:
                if first_failure_status is None:
                    first_failure_status = status
                log = str(evaluation.get("log") or "").strip()
                if log:
                    log_excerpts.append(f"[{status}] {log[-600:]}")

        all_passed = passed == total and total > 0
        log_excerpt = "\n".join(log_excerpts[:3])
        if error_text:
            log_excerpt = (error_text + "\n" + log_excerpt).strip()[:2000]

        metrics = {
            "passed_count": passed,
            "total_workloads": total,
            "feedback_workload_uuids": [str(w.get("uuid")) for w in workloads],
        }
        mean_speedup = sum(speedups) / len(speedups) if speedups else None
        return EvalResult(
            status="passed" if all_passed else str(first_failure_status or "failed"),
            latency_ms=(sum(latencies) / len(latencies) if latencies else None),
            reference_latency_ms=(
                sum(ref_latencies) / len(ref_latencies) if ref_latencies else None
            ),
            speedup_factor=mean_speedup,
            mean_vs_baseline_factor=mean_speedup,
            log_excerpt=log_excerpt[:2000],
            metrics=metrics,
        )

    def run_benchmark(
        self,
        *,
        solution: Any,
        config: Any = None,
        dump_traces: bool = False,
        round_num: int | None = None,
    ) -> EvalResult:
        """Feedback benchmark on the selected workload subset (the generators' API)."""
        if not isinstance(solution, TaskSolution):
            raise TypeError("run_benchmark expects a k_search.tasks.task_base.Solution")
        cfg: SolExecBenchEvalConfig = config or self._eval_config
        sol_solution = self.to_sol_solution(solution)
        workloads = list(self._selected_workloads)

        # Reuse cached reference latencies when every feedback workload already
        # has one; the driver then skips reference timing (benchmark_reference
        # only controls reference measurement, the candidate is always timed).
        use_cached_ref = cfg.benchmark_reference and all(
            str(w.get("uuid")) in self._ref_latency_by_uuid for w in workloads
        )
        effective_cfg = cfg
        if use_cached_ref:
            effective_cfg = SolExecBenchEvalConfig(
                warmup_runs=cfg.warmup_runs,
                iterations=cfg.iterations,
                benchmark_reference=False,
                correctness_only=cfg.correctness_only,
                eval_seed=cfg.eval_seed,
                timeout_seconds=cfg.timeout_seconds,
                per_workload_timeout_seconds=cfg.per_workload_timeout_seconds,
            )

        t0 = time.time()
        traces, error_text = self._run_sol_cli(
            sol_solution=sol_solution, workloads=workloads, config=effective_cfg
        )
        wall_s = time.time() - t0

        if cfg.benchmark_reference and not use_cached_ref:
            for trace in traces:
                perf = (trace.get("evaluation") or {}).get("performance") or {}
                ref = perf.get("reference_latency_ms")
                uuid = str((trace.get("workload") or {}).get("uuid"))
                if isinstance(ref, (int, float)) and float(ref) > 0 and uuid:
                    self._ref_latency_by_uuid[uuid] = float(ref)
            self._save_ref_latency_cache_to_disk()
        if use_cached_ref:
            for trace in traces:
                evaluation = trace.get("evaluation") or {}
                perf = evaluation.get("performance") or None
                uuid = str((trace.get("workload") or {}).get("uuid"))
                if not isinstance(perf, dict):
                    continue
                sol_lat = perf.get("latency_ms")
                ref = self._ref_latency_by_uuid.get(uuid)
                if (
                    evaluation.get("status") == "PASSED"
                    and isinstance(sol_lat, (int, float))
                    and float(sol_lat) > 0
                    and isinstance(ref, float)
                ):
                    perf["reference_latency_ms"] = ref
                    perf["speedup_factor"] = ref / float(sol_lat)

        er = self._eval_result_from_traces(
            traces=traces, workloads=workloads, error_text=error_text
        )
        er.metrics["cli_wall_clock_s"] = round(wall_s, 2)
        er.metrics["reference_latency_cached"] = bool(use_cached_ref)

        self._last_round_traces = traces
        self._last_round_passed_count = int(er.metrics.get("passed_count", 0))
        self._last_round_total_workloads = int(er.metrics.get("total_workloads", 0))
        self._last_round_trace_logs_for_prompt = self._feedback_logs_from_traces(
            traces, workloads
        )
        if error_text:
            self._last_round_trace_logs_for_prompt = (
                error_text + "\n" + self._last_round_trace_logs_for_prompt
            ).strip()
        self._last_round_summary_line = (
            f"[sol-task] {self.name}: {self._last_round_passed_count}/"
            f"{self._last_round_total_workloads} feedback workloads passed "
            f"({er.status}, wall {wall_s:.1f}s)"
        )
        print(self._last_round_summary_line, flush=True)

        # Feedback traces are tiny (a few KB per round) and double as the audit
        # log for every candidate, so always persist them.
        eval_dir = self._artifacts_eval_dir()
        if eval_dir is not None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            suffix = f"_r{round_num}" if round_num is not None else ""
            try:
                (eval_dir / f"feedback_traces{suffix}_{ts}.jsonl").write_text(
                    "".join(json.dumps(t) + "\n" for t in traces), encoding="utf-8"
                )
            except Exception:
                pass
        return er

    def run_final_evaluation(
        self,
        *,
        solutions: List[TaskSolution],
        config: Any = None,
        dump_traces: bool = False,
        workload_limit: int | None = None,
    ) -> Dict[str, Any]:
        """Evaluate solutions on ALL workloads through the same official CLI."""
        cfg: SolExecBenchEvalConfig = config or self._eval_config
        workloads = list(self._workload_records)
        if workload_limit:
            workloads = workloads[: int(workload_limit)]

        report: Dict[str, Any] = {
            "definition": self.name,
            "problem_dir": str(self.problem_dir),
            "workload_count": len(workloads),
            "solutions": {},
        }
        for solution in solutions:
            sol_name = str(getattr(solution, "name", "") or "solution")
            entry: Dict[str, Any] = {"passed": 0, "total": len(workloads)}
            try:
                sol_solution = self.to_sol_solution(solution)
                traces, error_text = self._run_sol_cli(
                    sol_solution=sol_solution, workloads=workloads, config=cfg
                )
                er = self._eval_result_from_traces(
                    traces=traces, workloads=workloads, error_text=error_text
                )
                entry.update(
                    {
                        "status": er.status,
                        "passed": er.metrics.get("passed_count", 0),
                        "mean_speedup": er.speedup_factor,
                        "mean_latency_ms": er.latency_ms,
                        "mean_reference_latency_ms": er.reference_latency_ms,
                        "log_excerpt": er.log_excerpt[:1000],
                    }
                )
                if dump_traces:
                    entry["traces"] = traces
            except Exception as e:
                entry.update({"status": f"CONVERSION_ERROR: {e}"})
            report["solutions"][sol_name] = entry
        return report
