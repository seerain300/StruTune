"""File-lock GPU pool for concurrent K-Search tasks (design: notes/GPU_POOL_DESIGN.md).

每个 GPU 一把 flock 文件锁：benchmark 子进程执行期间持有，进程死亡自动释放。
多个搜索进程并发运行（LLM 阶段不占卡），benchmark 到点时向池申请槽位、
阻塞排队；申请时做租户占用检测（显存上有陌生进程的卡暂时出池）。

用法（任务后端内）:
    from k_search.utils.gpu_pool import gpu_slot
    with gpu_slot("5,6") as gpu_id:      # KSEARCH_GPU_POOL 形式的卡号列表
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        ... 启动 benchmark 子进程并等待 ...

观测: /tmp/ksearch_gpu_pool/gpu<N>.lock 文件内容为持有者信息（pid/任务/时间）。
"""

from __future__ import annotations

import errno
import fcntl
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

LOCK_DIR = Path("/tmp/ksearch_gpu_pool")
EMPTY_THRESH_MIB = 200  # 卡上显存占用低于此值视为"空"
RECHECK_SEC = 30.0  # 池内全部不可用（租户占用）时的重扫间隔


def _lock_path(gpu: int) -> Path:
    return LOCK_DIR / f"gpu{gpu}.lock"


def _gpu_empty(gpu: int) -> bool:
    """该卡当前是否有陌生显存占用（不看我们自己的持锁 benchmark——持锁时本函数不再被调用）。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
    except Exception:
        return False  # 探测失败按"不可用"处理，宁可不抢
    if not out:
        return True
    for line in out.splitlines():
        line = line.strip()
        if line and line.isdigit() and int(line) >= EMPTY_THRESH_MIB:
            return False
    return True


def _write_holder(fp, gpu: int) -> None:
    try:
        task = os.environ.get("KSEARCH_POOL_TASK", os.environ.get("LLM_USAGE_RUN_ID", "?"))
        fp.seek(0)
        fp.truncate()
        fp.write(f"pid={os.getpid()} gpu={gpu} task={task} ts={time.strftime('%m%d %H:%M:%S')}\n")
        fp.flush()
    except Exception:
        pass


@contextmanager
def gpu_slot(pool_spec: str, *, acquire_timeout: float | None = None):
    """从池中申请一张空卡（阻塞排队），yield 卡号；退出时释放。

    pool_spec: 逗号分隔卡号，如 "5,6"。
    acquire_timeout: None=无限等待；超时抛 TimeoutError。
    """
    pool = [int(x) for x in str(pool_spec).split(",") if str(x).strip().isdigit()]
    if not pool:
        raise ValueError(f"empty GPU pool: {pool_spec!r}")
    LOCK_DIR.mkdir(parents=True, exist_ok=True)

    deadline = time.time() + acquire_timeout if acquire_timeout else None
    # 轮转起点错开，让等待者天然分散到不同卡
    idx = int(time.time() * 10 + os.getpid()) % len(pool)

    while True:
        for step in range(len(pool)):
            gpu = pool[(idx + step) % len(pool)]
            if not _gpu_empty(gpu):
                continue  # 租户占用，跳过
            path = _lock_path(gpu)
            fh = open(path, "a+")
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as e:
                fh.close()
                if e.errno in (errno.EACCES, errno.EAGAIN):
                    continue  # 别的任务持有，试下一张
                raise
            try:
                _write_holder(fh, gpu)
                yield gpu
                return
            finally:
                try:
                    fh.seek(0); fh.truncate(); fh.write("free\n"); fh.flush()
                except Exception:
                    pass
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                fh.close()
        if deadline and time.time() > deadline:
            raise TimeoutError(f"gpu_slot: no GPU available from pool {pool} in {acquire_timeout}s")
        time.sleep(RECHECK_SEC)


def pool_status() -> dict:
    """观测：每张池卡的锁状态。"""
    out = {}
    if not LOCK_DIR.exists():
        return out
    for p in sorted(LOCK_DIR.glob("gpu*.lock")):
        try:
            fh = open(p, "a+")
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                held = False
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                held = True
            info = ""
            if held:
                fh.seek(0)
                info = fh.read().strip()
            fh.close()
            out[p.stem] = {"held": held, "holder": info or None}
        except Exception:
            continue
    return out


if __name__ == "__main__":
    import json as _json

    print(_json.dumps(pool_status(), indent=2, ensure_ascii=False))
