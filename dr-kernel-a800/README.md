# dr-kernel-a800 — drkernel-8b 推理评测流水线实验归档（A800）

> 实验线：以官方 KernelBench 风格合同 + Triton-only 约束 + STTS 测试时扩展协议，
> 在自建 30 题（flashinfer-test 10 题 + SOL-ExecBench L1/L2 各 10 题）上评测
> `hkust-nlp/drkernel-8b` 的 kernel 生成能力上限。
> 运行时间：2026-09-18 ~ 2026-09-20；硬件：NVIDIA A800-SXM4-80GB ×8。

## 一、实验协议

**提示词合同**（`scripts/drkernel_kbstyle_pass1.py`）：官方模型卡 1-shot KernelBench 模板
（Model/get_inputs → ModelNew，输出 markdown codeblock），追加两段定制：
① 全部 workload 的 axes 清单（要求对全部配置正确）；② Triton-only 硬约束
（所有计算必须在自己写的 @triton.jit kernel 中；host 侧只允许 shape/stride/分配/启动）。

**STTS 协议**（`scripts/kbstyle_stts.py`，忠实移植官方 KernelGYM
`drkernel-14b-maxturns5-maxiter10.sh` / `openai_async_engine_multi_iter.py`）：
- 8 采样/题；每迭代段最多 5 个用户轮；最多 10 个迭代；段间保留 reward 总和最大的
  连续 4 轮窗口（best-window）；无耐心早停（跑满）；报告 best-of-history
- 采样：T=1.0、top_p=0.95、max_tokens=8192、seed=20260918、stop=["<|im_end|>"]，
  双 vLLM 服务器轮询（同权重同 seed，输出一致）
- 每轮流程：生成 → AST 合规检测（decoy kernel / host 侧 torch 计算白名单）→
  不合规不入评测、违规清单进反馈 → 合规则全量 workload 评测（缓存参考输出 +
  候选计时 warmup 3 + 10 iters）→ 结构化反馈（X/Y correct + geomean + 失败
  workload 的 axes/错误摘要）
- reward = 0.3×编译 + 0.4×正确率 + 0.3×min(geomean, 3.0)/3.0
- **终局口径**：每题取 8×全部轮次中 reward 最高的历史最优解，官方评测器
  （`evaluators/evaluate.py` / `evaluate_sol.py`）全新复评：全 workload、
  参考重计时、warmup 3 + 100 iters

**批次二（基线）**：`tasks_v2/` 为同协议的 3 轮版（8 采样 × 最多 3 轮反馈，
`kbstyle_campaign.py`），用于观察迭代深度的边际效应。

**与官方 KernelGYM 的已记录分歧**（复现对照时注意）：
① 历史重建保留任务 prompt（官方只保留系统提示+窗口）；
② 无 env-done 段提前结束（我方恒跑满段预算，方向上多探索）；
③ prompt 预算用消息级压缩（~17k tok）而非官方 20480 token 左截断；
④ 防作弊用 AST 静态检测而非沙箱 profiling（coverage 加分官方评测默认关闭，
   与本实验一致，见 `batch_logs/` 内审计记录）。
协议偏离：56/240 采样实际 51–55 轮（中断重启的段预算 bug，已修复，多发轮次
只增探索不虚报成绩）。

## 二、结果总表

见 `summary/results_table.md`（人类可读）与 `.csv`（程序可读）——由
`gen_tables.py` 从原始 JSON 生成，未手抄。摘要（stts 批次）：

- **又对又快（>1x）：3/30** — rmsnorm 3.30x、SOL/L1/053 3.22x、SOL/L1/008 2.19x
- 正确但慢：4/30 — 058 0.65x、gemm 0.55x、mla_paged_decode 0.21x、L2/030 0.14x
- 未解出：23/30（8 采样 × 47 轮探索均无全对解）
- 迭代深度效应（同题 3轮 → 10迭代）：053 0.10x→3.22x、058 0.01x→0.65x、
  008 1.71x→2.19x；pass@1 在"正确性可达"题上全部 ≥0.62

## 三、口径与硬件绑定

- **所有 speedup 均为 A800 实测**，与 H100 不可比（参考实现延迟硬件绑定）；
  引用须标注硬件
- 轮级反馈口径（缓存参考延迟 + 10 iters）仅用于轨迹内选择与表格 best 列，
  与终局复评口径不可混用；两列并列仅为可追溯
- 参考延迟缓存（refcache）与硬件绑定，**未收录本归档**，复现需在新硬件重跑
  precompute（`scripts/kbstyle_fib_eval.py --mode precompute`）

## 四、目录导航

```
dr-kernel-a800/
├── README.md                  ← 本文件
├── task_plan.json             ← 30 题定义（含各题 PyTorch 参考实现/workload 路径）
├── summary/
│   ├── results_table.md/.csv  ← 总表（两批次并列，未解题标注"未解出"）
│   └── best_solutions/        ← 终局复评有效解（7 个，文件头含来源采样/轮次/reward）
├── tasks/<benchmark>/<题名>/          ← stts 批次原始产物（30 题；每题仅保留
│                                最优轨迹的轮目录，其余 7 条轨迹存
│                                state.json 摘要——pass@1 可复核，
│                                非最优轨迹无原始回复）
│   ├── prompt.txt / state.json / summary.json / best_solution.py / final/
│   └── s<k>t<n>/              ← 每轮：response.txt（模型原话，内含完整 ModelNew 代码）/ solution.py（+run 包装）/
│                                 evaluation.json（per-workload 明细）/ evaluation.log
├── tasks_v2/<benchmark>/<题名>/       ← 3 轮基线批次（同结构，s<k>t<n> 为轮目录）
├── batch_logs/
│   ├── campaign.log / monitor.log / results.jsonl / tokens.jsonl   ← stts 批次
│   └── stts3turn/{campaign.log,results.jsonl,scan_results.jsonl}   ← 3轮批次
├── scripts/                   ← 流水线全部脚本（8 个，含双 vLLM 启动/监控）
└── evaluators/                ← 官方评测器 evaluate.py / evaluate_sol.py
```

查"某题某轮模型说了什么"：`tasks/<题>/s<k>t<n>/response.txt`；
查"评测判了什么"：同目录 `evaluation.json`（per-workload：status/ref_ms/sol_ms/
speedup/axes）；查"该题最终成绩"：`tasks/<题>/final/evaluation.json`。

## 五、关键工程事项（复现必读）

1. **vLLM 停止符**：模型 eos=151643 而 chat 模板以 `<|im_end|>`(151645) 收尾；
   启动须带 `--override-generation-config '{"eos_token_id":[151643,151645]}'`
   （`scripts/start_drkernel_gpu*.sh` 已内置），否则模型以空行填满 8192 token。
2. **评测模式名**：`kbstyle_fib_eval.py --mode` 仅接受 precompute/feedback/screen；
   campaign 侧 'full' 已在代码内映射为 feedback（历史坑，勿回退）。
3. **段恢复预算**：中断重启会重发在飞段的轮次预算（已修：seg_done_turns 持久化）；
   归档数据含此 bug 的 56 个 51–55 轮采样。
4. **GPU 池防污染**：评测按 nvidia-smi 计算进程探活，自动避开他人任务；
   本实验触发让卡等待多次（见 batch_logs/campaign.log "GPU pool: all busy" 行）。
5. 已知数据噪声：gdn_decode s7t17 的 evaluation.log 为 Triton 编译器 loc 行
   无限重复（原始 6.6GB），归档时截断至前 2MB；对应 evaluation.json 完整保留。
6. token 总账：stts 批次 11,172 请求 / prompt 110.3M + completion 91.5M
   （batch_logs/tokens.jsonl 逐请求可审计）。

## 六、复现入口

```bash
# 1) 双 vLLM（GPU/端口按机器改）：bash scripts/start_drkernel_gpu7_kbstyle.sh ...
# 2) 参考预缓存（硬件绑定，必须重跑）：
#    kbstyle_fib_eval.py --mode precompute --definition ... --cache-dir refcache/fib/<题> ...
# 3) 3 轮基线：kbstyle_campaign.py --phase both --max-parallel 6 --concurrent-tasks 3
# 4) STTS：kbstyle_stts.py --samples 8 --iterations 10 --patience 0 \
#        --concurrent-tasks 4 --gpus 0,1,2,3,4,5 --max-parallel 6
```
