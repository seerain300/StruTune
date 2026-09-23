#!/usr/bin/env python3
"""Resident keeper that re-acquires GPU occupancy holders after the shared
evaluation/profiling lease goes idle.

配合 evaluate_candidate.py 的计数式共享占卡租约（occupancy_lease）工作，实现
"队列忙时保持让位、队列空闲后占回"的批量占卡策略：

  - 各评测/剖析进程通过 occupancy_lease 共享让位：第一个进入者停掉持有器，
    最后一个退出者只登记空闲时间戳（不立即占回，避免连续评测之间停/启切换）；
  - 本守护进程周期扫描各卡的共享租约状态：
      * 清理已死亡进程的计数（评测进程被 timer 窗口击杀等场景，防止泄漏）；
      * 当某卡租约计数归零、且空闲超过 KDA_OCCUPANCY_IDLE_SECONDS（默认 90 秒）、
        且该卡本批开始前确实有持有器在跑而现在没跑 —— 重新拉起持有器占回。
  - 只处理自己登记过的卡（状态文件里的 holder_was_running 标记）；外部手动
    stop 的卡不会被擅自拉起。

启动（与 campaign driver 同生命周期，建议一并 nohup）：
    KDA_ALLOWED_GPUS=3 python3 holder_keeper.py --interval 10 [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_candidate as ec


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def keeper_once(gpus: list[int], idle_seconds: float, dry_run: bool = False) -> list[dict]:
    """扫描一轮全部受管卡，返回本轮动作记录。

    持有器生命周期全部经由协调状态（coordination flock 保护）管理，状态含义：
      holders 非空           —— 评测/剖析租约进行中（持有器应保持停止）；
      holder_was_running     —— 本批开始前持有器在跑，当前欠一次占回；
      idle_since             —— 最后一个租约退出的时刻（欠账但未到占回阈值）；
      holder_restarting_since—— 正在（重新）获取持有器的过渡态：获取期间新租约
                                会在协调锁上排队等待，不会与分配中的持有器冲突
                                （这正是"占卡程序占卡过程中和队列新任务冲突"的治理点）；
      holder_requested       —— 运营者请求占卡（--request-holder 设置），下一轮
                                在无租约时执行，替代绕过状态机的手动 gpu_occupancy start。
    """
    hostname = ec.socket.gethostname()
    actions: list[dict] = []
    for gpu_index in gpus:
        state_path = ec.occupancy_coordination_path(hostname, gpu_index)
        with ec.coordination_flock(hostname, gpu_index):
            state = ec.load_coordination_state(state_path)  # 已剔除死 pid
            if state["holders"]:
                continue  # 仍有租约在用：持有器必须保持停止
            restarting_since = parse_timestamp(state.get("holder_restarting_since"))
            if restarting_since is not None:
                if (datetime.now().astimezone() - restarting_since).total_seconds() > 120:
                    # 获取过程卡死（超过 2 分钟）：清过渡态，回到欠账重试
                    state.pop("holder_restarting_since", None)
                    state["idle_since"] = now_iso()
                    ec.write_coordination_state(state_path, state)
                    actions.append({"gpu": gpu_index, "action": "restart-stuck-cleared"})
                elif ec.occupancy_holder_running(gpu_index):
                    # 获取完成：清过渡态、欠账与请求标记
                    state.pop("holder_restarting_since", None)
                    state["holder_was_running"] = False
                    state.pop("idle_since", None)
                    state.pop("holder_requested", None)
                    ec.write_coordination_state(state_path, state)
                    actions.append({"gpu": gpu_index, "action": "holder-restarted",
                                    "time": now_iso()})
                continue  # 过渡态中不做其他动作
            want_restart = (
                (state.get("holder_was_running") or state.get("holder_requested"))
                and not ec.occupancy_holder_running(gpu_index)
            )
            if not want_restart:
                continue
            if ec.gpu_occupancy is None:
                continue
            idle_since = parse_timestamp(state.get("idle_since"))
            is_request = bool(state.get("holder_requested"))
            if idle_since is None:
                state["idle_since"] = now_iso()
                ec.write_coordination_state(state_path, state)
                continue
            if not is_request and (datetime.now().astimezone() - idle_since).total_seconds() < idle_seconds:
                continue  # 欠账占回需等空闲阈值；显式请求则立即执行
            if dry_run:
                actions.append({"gpu": gpu_index, "action": "would-restart-holder"})
                continue
            # 进入"获取中"过渡态（持锁登记后才开始分配，新租约会在锁上排队）
            state["holder_restarting_since"] = now_iso()
            ec.write_coordination_state(state_path, state)
            with ec.gpu_occupancy._gpu_lock(gpu_index):
                ec.gpu_occupancy._start_locked(gpu_index)
            # 短暂验证：活的在下一轮扫描确认（holder_restarting 过渡态中完成闭环），
            # 秒死（解释器缺 torch 等）由 120 秒卡死检测兜底重试。
            actions.append({"gpu": gpu_index, "action": "holder-restart-initiated",
                            "time": now_iso()})
    return actions


def request_holder(gpu_index: int) -> None:
    """运营者安全的占卡请求：登记 holder_requested，由守护循环在无租约时执行。

    替代直接运行 gpu_occupancy.py start——后者绕过协调状态，可能与进行中的
    评测租约冲突（分配中的持有器会被外部进程监控判为干扰进程，作废评测）。
    """
    hostname = ec.socket.gethostname()
    state_path = ec.occupancy_coordination_path(hostname, gpu_index)
    with ec.coordination_flock(hostname, gpu_index):
        state = ec.load_coordination_state(state_path)
        state["holder_requested"] = True
        ec.write_coordination_state(state_path, state)
    print(f"holder requested on gpu{gpu_index}; keeper will acquire it when no lease is active",
          flush=True)


def release_holder(gpu_index: int) -> None:
    """运营者安全的放卡：停持有器并清除全部占回欠账/请求标记。

    直接运行 gpu_occupancy.py stop 的隐患：协调状态里的欠账
    （holder_was_running / holder_requested）依然存在，keeper 会在空闲阈值后
    违背放卡意图地占回。本命令先在协调锁内清账，再停持有器，语义完整。
    若评测租约正在进行（计数非零），拒绝执行以免干扰计时。
    """
    hostname = ec.socket.gethostname()
    state_path = ec.occupancy_coordination_path(hostname, gpu_index)
    with ec.coordination_flock(hostname, gpu_index):
        state = ec.load_coordination_state(state_path)
        if state["holders"]:
            print(f"gpu{gpu_index}: evaluation/profiling lease active; refusing to release "
                  f"({len(state['holders'])} holder(s)) — retry when idle", flush=True)
            return
        state["holder_was_running"] = False
        state["holder_requested"] = False
        state.pop("idle_since", None)
        state.pop("holder_restarting_since", None)
        ec.write_coordination_state(state_path, state)
        if ec.gpu_occupancy is not None and ec.occupancy_holder_running(gpu_index):
            with ec.gpu_occupancy._gpu_lock(gpu_index):
                ec.gpu_occupancy._stop_locked(gpu_index)
    print(f"gpu{gpu_index}: holder stopped and re-acquire debt cleared", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="",
                        help="comma-separated GPU indices; defaults to $KDA_ALLOWED_GPUS")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--idle-seconds", type=float,
                        default=float(os.environ.get("KDA_OCCUPANCY_IDLE_SECONDS", "30")))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--request-holder", type=int, default=None, metavar="GPU",
                        help="one-shot: register a holder request for GPU and exit "
                             "(safe replacement for manually running gpu_occupancy.py start)")
    parser.add_argument("--release-holder", type=int, default=None, metavar="GPU",
                        help="one-shot: stop the holder on GPU and clear the re-acquire "
                             "debt so the keeper will not re-occupy against operator intent")
    args = parser.parse_args()

    if args.request_holder is not None:
        request_holder(args.request_holder)
        return 0
    if args.release_holder is not None:
        release_holder(args.release_holder)
        return 0

    raw = args.gpus or os.environ.get("KDA_ALLOWED_GPUS", "")
    gpus = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not gpus:
        raise SystemExit("no GPUs to manage: pass --gpus or set KDA_ALLOWED_GPUS")

    # 持有器 fill_vram.py 依赖 torch；本守护以系统 python3 运行，
    # 必须为重启子进程指定带 torch 的解释器（与 evaluate_candidate 的做法一致）。
    os.environ.setdefault("GPU_OCCUPANCY_PYTHON", str(ec.FLASHINFER_PYTHON))

    print(f"holder_keeper: gpus={gpus} interval={args.interval}s "
          f"idle-threshold={args.idle_seconds}s", flush=True)
    while True:
        try:
            for action in keeper_once(gpus, args.idle_seconds, args.dry_run):
                print(json.dumps(action, ensure_ascii=False), flush=True)
        except Exception as error:  # 守护进程不因单轮异常退出
            print(f"[holder_keeper] scan error: {error}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
