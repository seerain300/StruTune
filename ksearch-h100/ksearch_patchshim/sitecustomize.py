"""sitecustomize shim: 在每个 Python 子进程启动时给 flashinfer_bench 的反馈
正确性检查打 -inf 哨兵语义补丁（对齐 ziming evaluate.py 的 -inf==-inf 特判）。

背景：gqa_paged_prefill 等题的 reference 输出（LSE）合法地含 -inf（空 causal
前缀的 logsumexp），原版 check_correctness 对任何含 inf 的候选直接判死，导致
正确解（含 reference 自身）永远过不了反馈。isolated runner 用 spawn 子进程跑
评测，本进程的 monkey-patch 不生效，故通过 PYTHONPATH 前置本目录 + sitecustomize
自动加载来注入。详见 K-Search/k_search/tasks/flashinfer_bench_task.py 的
patch_feedback_checker_inf_sentinels。幂等；对输出全有限的 definition 无行为
变化；flashinfer_bench 不可导入的环境（如 SOL venv）静默跳过。
"""

try:
    from k_search.tasks.flashinfer_bench_task import (
        patch_feedback_checker_inf_sentinels,
    )

    patch_feedback_checker_inf_sentinels()
except Exception:
    pass
