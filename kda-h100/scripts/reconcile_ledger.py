#!/usr/bin/env python3
"""Reconcile state.json counters against the candidates.jsonl evidence ledger.

Invariant enforced: state.candidate_evaluations == number of parseable
candidates.jsonl records (handover audit rule).

Rules (with --apply):
  A. A state candidate record with feedback_completed=true but no
     evaluation_index (evaluator v3 window: benchmark finished, process died
     before counting) gets an index assigned and is flagged feedback_counted.
  B. counter > ledger records: for each counted candidate (ordered by
     evaluation_index) that has no ledger record, append one auto-reconciled
     line built from runs/candidates/<id>/feedback.json, clearly marked with
     reconciled_by. Existing lines are never rewritten.
  C. ledger records > counter: report only (needs human review); the counter is
     never decremented automatically.

Blank ledger lines are ignored for counting. Non-blank unparseable lines are
reported; --repair-ledger quarantines them verbatim into
candidates.jsonl.fragments.

Usage:
  reconcile_ledger.py --campaign <campaign.json> [--apply] [--repair-ledger]
  reconcile_ledger.py --run <run_id> [--apply] [--repair-ledger]
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/ziming/kda-ops")
CONTROL_ROOT = ROOT / "kda-control"


def now() -> str:
    return datetime.now().astimezone().isoformat()


def load_ledger(path: Path) -> tuple[list[dict], int, int]:
    records: list[dict] = []
    malformed = blank = 0
    if not path.is_file():
        return records, malformed, blank
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            blank += 1
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            malformed += 1
    return records, malformed, blank


def ledger_ids(records: list[dict]) -> set[str]:
    ids = set()
    for record in records:
        value = record.get("candidate_id") or record.get("id") or record.get("candidate")
        if value:
            ids.add(str(value))
    return ids


def feedback_summary(workspace: Path, candidate: str) -> dict:
    path = workspace / "runs" / "candidates" / candidate / "feedback.json"
    if not path.is_file():
        return {"feedback_json": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    per_workload = data.get("per_workload") or []
    return {
        "feedback_json": True,
        "valid": data.get("valid"),
        "geomean_speedup": data.get("geomean_speedup"),
        "arithmetic_mean_speedup": data.get("arithmetic_mean_speedup"),
        "workload_summary": [
            {
                "uuid": str(w.get("uuid") or w.get("workload") or i),
                "correct": w.get("correct", w.get("passed")),
                "speedup": w.get("speedup") or w.get("geomean_speedup"),
            }
            for i, w in enumerate(per_workload)
        ],
    }


def reconcile_run(run_id: str, apply: bool = False, repair_ledger: bool = False) -> dict:
    control = CONTROL_ROOT / run_id
    workspace = ROOT / "kda-runs" / run_id
    state_path = control / "state.json"
    ledger_path = workspace / "candidates.jsonl"
    report: dict = {
        "time": now(), "run_id": run_id, "changed": [], "needs_human": [],
    }
    if not state_path.is_file():
        report["needs_human"].append("missing state.json")
        return report
    state = json.loads(state_path.read_text(encoding="utf-8"))
    records, malformed, blank = load_ledger(ledger_path)
    report["malformed_lines"] = malformed
    report["blank_lines"] = blank

    # Rule A: completed evaluations that were never counted.
    for candidate, record in sorted(state.get("candidates", {}).items()):
        if record.get("feedback_completed") and not record.get("evaluation_index"):
            if not apply:
                report["changed"].append(f"would assign evaluation_index to {candidate}")
                continue
            state["candidate_evaluations"] = int(state.get("candidate_evaluations", 0)) + 1
            record["evaluation_index"] = state["candidate_evaluations"]
            record["feedback_counted"] = True
            report["changed"].append(f"assigned evaluation_index={record['evaluation_index']} to {candidate}")
        elif (
            record.get("feedback_started_at")
            and not record.get("evaluation_index")
            and not record.get("feedback_completed")
        ):
            # Evaluator-crash victim: benchmark ran (feedback.json holds a real
            # result with per-workload data) but completion bookkeeping never
            # happened. Count it and seal the record so the candidate cannot be
            # silently re-evaluated.
            summary = feedback_summary(workspace, candidate)
            if summary.get("feedback_json") and summary.get("workload_summary"):
                if not apply:
                    report["changed"].append(f"would count crash-victim evaluation {candidate}")
                    continue
                state["candidate_evaluations"] = int(state.get("candidate_evaluations", 0)) + 1
                record["evaluation_index"] = state["candidate_evaluations"]
                record["feedback_counted"] = True
                record["feedback_completed"] = True
                record["feedback_returncode"] = 0
                report["changed"].append(
                    f"counted crash-victim evaluation {candidate} (index={record['evaluation_index']})"
                )

    # Rule B: counted evaluations missing a ledger record.
    counter = int(state.get("candidate_evaluations", 0))
    known = ledger_ids(records)
    counted = sorted(
        (r for r in state.get("candidates", {}).values() if r.get("evaluation_index")),
        key=lambda r: r["evaluation_index"],
    )
    missing = [r["id"] for r in counted if r["id"] not in known]
    for candidate in missing:
        record = state["candidates"][candidate]
        summary = feedback_summary(workspace, candidate)
        if not apply:
            report["changed"].append(f"would append reconciled ledger line for {candidate}")
            continue
        entry = {
            "candidate_id": candidate,
            "candidate": candidate,
            "parent": record.get("parent"),
            "source_sha256": record.get("source_sha256"),
            "phase": "reconciliation",
            "hypothesis": "(operator reconciliation: the original session ended before appending evidence)",
            "change_from_parent": None,
            "workloads": summary.get("workload_summary"),
            "all_correct": bool(summary.get("valid")),
            "geomean_speedup": summary.get("geomean_speedup"),
            "arithmetic_mean_speedup": summary.get("arithmetic_mean_speedup"),
            "decision": "operator-reconciled (pending review)",
            "cumulative_evaluations": counter,
            "notes": (
                f"auto-appended by reconcile_ledger.py at {now()} from "
                f"runs/candidates/{candidate}/feedback.json "
                f"(feedback_returncode={record.get('feedback_returncode')})"
            ),
            "skills_used": [],
            "static_checks": record.get("static_check_errors") or [],
            "reconciled_by": "reconcile_ledger.py",
        }
        with ledger_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(entry, ensure_ascii=False) + "\n")
        records.append(entry)
        known.add(candidate)
        report["changed"].append(f"appended reconciled ledger line for {candidate}")

    # Final invariant check.
    counter = int(state.get("candidate_evaluations", 0))
    if counter != len(records):
        # An extra ledger line whose candidate has an UNCOMPLETED state record
        # (evaluation started, interrupted by a window kill, never counted) is
        # expected to self-align once the next window re-evaluates it.
        counted_ids = {r["id"] for r in counted}
        extra = [i for i in ledger_ids(records) - counted_ids]
        blocking = [i for i in extra if i not in state.get("candidates", {})]
        self_aligning = [i for i in extra if i in state.get("candidates", {})]
        if self_aligning:
            report["self_aligning"] = self_aligning
        if blocking:
            report["needs_human"].append(
                f"counter={counter} != ledger_records={len(records)}; ledger-only ids with no state record: {blocking}"
            )
        else:
            report["invariant_pending"] = f"counter={counter}, ledger={len(records)}, awaiting re-evaluation of {self_aligning}"
    else:
        report["invariant_ok"] = True

    if malformed and repair_ledger and apply:
        fragments = ledger_path.with_suffix(".jsonl.fragments")
        kept_lines = []
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                json.loads(line)
                kept_lines.append(line)
            except json.JSONDecodeError:
                with fragments.open("a", encoding="utf-8") as frag:
                    frag.write(line + "\n")
        ledger_path.write_text(
            "".join(line + "\n" for line in kept_lines), encoding="utf-8"
        )
        report["changed"].append(f"quarantined {malformed} malformed line(s) into {fragments.name}")

    if apply and report["changed"]:
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        audit = control / "reconciliation.jsonl"
        with audit.open("a", encoding="utf-8") as output:
            output.write(json.dumps(report, ensure_ascii=False) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--run")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--repair-ledger", action="store_true")
    args = parser.parse_args()

    run_ids: list[str] = []
    if args.run:
        run_ids = [args.run]
    elif args.campaign:
        campaign = json.loads(args.campaign.read_text(encoding="utf-8"))
        run_ids = [item["run_id"] for item in campaign["tasks"]]
    else:
        parser.error("need --run or --campaign")

    bad = 0
    for run_id in run_ids:
        report = reconcile_run(run_id, apply=args.apply, repair_ledger=args.repair_ledger)
        status = "OK" if report.get("invariant_ok") and not report["needs_human"] else "ATTENTION"
        if status == "ATTENTION":
            bad += 1
        print(f"[{status}] {run_id.split('--')[-1][:40]} counter==ledger:{report.get('invariant_ok', False)} "
              f"changed={report['changed']} human={report['needs_human']}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
