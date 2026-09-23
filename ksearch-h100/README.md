# K-Search H100 实验产物归档

> LLM 驱动 GPU kernel 优化（K-Search world-model 完整版）在 H100 上的 30 题复现实验。
> 整理时间：2026-09-23。结构标准见仓库根 `whr/实验产物整理与上传规范.md`（样板：dr-kernel-h100）。

## 1. 实验概况

| 项 | 值 |
|---|---|
| 方法 | K-Search world-model 完整版（节点树探索 + 解库血缘），seed0 |
| LLM | GPT-5.6-Sol @ llmapi.isrc.ac.cn/v1（OpenAI 兼容网关） |
| 硬件 | NVIDIA H100 80GB HBM3（共享机，多战役+多实验线并行的真实环境） |
| 题集 | FlashInfer-test 10 题 / 426 workloads + SOL-ExecBench L1 10 题 + L2 10 题 = **30 题 / 738 workloads** |
| 搜索预算 | 每题 100 轮（按**完成评测数**计数，非启动数）；WM 节点 20 × attempt 5 |
| 反馈评测 | 全量 workload；FI: warmup3/iters10-20/trials5；SOL: warmup10/iters10-20 |
| 终评 | 全量 workload，warmup3 + **100 iterations**，参考与候选同进程成对计时（无缓存），每 workload speedup = ref_ms/sol_ms，题目得分 = 几何平均 |
| 守卫 | BANNED 清单（matmul/mm/bmm/addmm/einsum/F.linear/F.conv*/fft/cumsum/sort/topk/unique/@torch.compile）；退出解静态扫描，命中即违规留档不终评 |

## 2. 结果速览

- **有效终评 21 题：geomean-of-geomean = 21.10x**（标准口径 20 题 = 18.72x，
  唯一非标题 gdn_prefill 见口径注记）
- **违规留档 9 题**（L1 3 + L2 6）：退出解含 BANNED torch 调用——守卫失效集中在
  conv/FFT/批量矩阵类题目（详见 §5 与 docs/L2_TORCH_GUARD_AUDIT.md）
- Token 消耗：30 题有效 **93.4M**（输入 72.2M / 输出 22.9M；输入 43% 命中网关
  prompt-cache——这是算法真实特性：WM 稳态下 prompt 前缀稳定）

完整逐题表（含 token 拆分、轮次、注记）：`summary/results_table.md`（CSV 同目录）。
逐 workload 计时明细：`summary/best_timing_detail.md/.csv`。

## 3. 目录导航

```
ksearch-h100/
├── README.md                        ← 本文件
├── summary/                         ← 30 秒查询层（最重要）
│   ├── results_table.md/.csv        ← 30 题总表（终评/违规/轮次/token/注记）
│   ├── best_timing_detail.md/.csv   ← 21 题 × 逐 workload ref_ms/sol_ms/speedup
│   └── best_solutions/<题>.py       ← 21 个有效终评解（文件头含来源/指标/token）
├── tasks/<题目>/<批次>/             ← 原始产物（批次=formal_h100 / formal_sol_h100 / formal_solL2_h100）
│   ├── ksearch-artifacts/<题>/
│   │   ├── rounds/round_NNN_{passed,failed}.py      ← 每轮候选代码（含失败轮）
│   │   ├── solutions/<题>/*.json    ← 保存解（JSON 含代码+评测结果+血缘）
│   │   ├── world_model/{world_model,solution_db}.json ← WM 状态与解库账本
│   │   └── eval/<题>/feedback_traces_rN_*.jsonl      ← 每轮评测明细（per-workload）
│   ├── final/                       ← 终局复评（unified/{evaluation|performance|candidate}.json + 解代码）
│   ├── usage.jsonl                  ← 该题 LLM 逐调用记账（tokens 含 cached 拆分）
│   ├── campaign_stdout.log          ← 该题全程日志（轮次横幅/Round summary/污染重试）
│   └── DONE / exit_code             ← 收官标记与退出码
├── batch_logs/                      ← 批级日志（campaign 进度、supervisor 重启记录）
├── scripts/                         ← 流水线脚本（campaign/评测/归档，含 build_h100_archive.py）
├── evaluators/                      ← FI 评测器 evaluate.py（与 ziming test/scripts 逐字节同源）
├── ksearch_patchshim/               ← -inf 哨兵补丁（sitecustomize 注入评测子进程）
└── docs/                            ← 协议、RUNBOOK、审计报告、事故复盘、token 总表等
```

查"某题某轮模型生成了什么/评测判了什么"：
`tasks/<题>/<批次>/ksearch-artifacts/<题>/rounds/round_NNN.py`（代码）+
`eval/<题>/feedback_traces_rN_*.jsonl`（评测）+ `campaign_stdout.log`（轮次上下文）。

## 4. 口径说明（引用数字前必读）

1. **硬件绑定**：所有加速比基于 H100（sm_90）；与 A800 数字不可混用。协议原文
   （docs/protocol.md）写的是 A800，本实验是同协议的 H100 移植版。
2. **终评口径**：warmup3 + 100 iterations，全量 workload，L2 cache 每迭代清空、
   输入张量每迭代克隆（flashinfer-bench `do_bench` 语义）。SOL 题走 SOL 官方 CLI
   （iterations 100）。
3. **gdn_prefill 非标口径（†）**：warmup3 + **20 iterations** + 参考延迟走磁盘缓存
   （该题全量参考单遍 521 分钟，实时重测不经济）。候选侧仍实时计时，100/100 全过。
   剔除该题后 20 题 geomean = 18.72x。
4. **搜索期反馈数字 ≠ 终评数字**：反馈用 10-20 迭代抽样口径（快），终评恒为
   100 迭代全量；两者差 2-4x 属正常（冷启动/L2 效应摊薄）。
5. **gemm_n4096_k4096 的 0.47x 是真实结果**：r1-59 自研全部失败，退出解 = 调与
   reference 同款的 cuBLAS matmul；该题处于方法能力边界（H100 cuBLAS 不可超）。
6. **043 的 3.20x 是搜索期 best-round 干净解复评**：退出解违规（torch.compile+
   cumsum+bmm），终评用的是 100 轮内最佳合规解。
7. **违规留档题的搜索记录全部保留**（搜索真实发生，token 照记），仅不进入终评统计。

## 5. 关键方法学发现

- **守卫失效谱系**（BANNED 静态扫描，LLM 提示词级约束）：L2 违规率 8/10 ≫ L1 3/10 ≫
  FI 0/10。失效集中在 conv/FFT/批量矩阵（cuDNN/cuFFT/cuBLAS 委托类）；FI 注意力题
  全部干净。说明提示词级守卫对"库函数能直接完成核心计算"的题目无效——模型会理性
  选择调库而非自研。审计逐题结论见 docs/L2_TORCH_GUARD_AUDIT.md 与
  docs/reports/k-search_30t_0917.md。
- **world-model 的 prompt-cache 红利**：稳态搜索时 prompt 前缀（协议+WM 状态）稳定，
  输入 token 43% 命中网关缓存，实际成本远低于裸计数。
- **搜索瓶颈实测**（三 FI 题日志拆解）：每轮 ~39% 时间在 LLM 生成（4-5k token/轮
  ≈ 2-3min），~61% 在评测窗口排队/污染重试；纯 GPU 计算占比极小。

## 6. 影响复现的工程事项

1. **DONE ≠ 100 轮**：任务 rc=0 退出且写 DONE 不代表预算跑满（WM 节点/attempt 上限
   会提前收尾）。重启续跑前必须 `grep -c "Round summary"` 核实真实轮次；本实验
   gqa_paged_prefill 曾 96 轮带 DONE 退出，已补跑至 106。
2. **同一张卡的任务必须同一 campaign 实例拉起**：污染检测器按 KSEARCH_OWNER_TAG
   互认"自己人"；两个 campaign 实例的 tag 不同，同卡时互相把对方的评测窗口判为
   陌生租户污染而无限丢弃重试（2026-09-23 事故：两题互相卡死 1 小时零进展）。
3. **同卡评测窗口的隐性竞态**：`stranger_ok` 先查 compute-apps 再查总显存，两次
   快照间自家评测 worker 新建上下文 >1GB 会被误判"不可见陌生占用"。解法：
   `KSEARCH_INVISIBLE_ALLOW_MB=6000`（容忍 6GB）或跨卡分离任务。
4. **共享机礼让协议**：fill_vram 持有器在评测间隙占住显存（防同事抢卡），窗口开启
   时自动让位；陌生进程上卡 → 窗口丢弃重试（不消耗轮次预算）。全实验累计 400+ 次
   STRANGER 丢弃，均无预算损耗。
5. **LLM 网关静默挂死**：无超时的 LLM 调用会永久阻塞任务（本实验 6+ 次）。已加
   300s 客户端超时（ksearch-token-run.py）+ 看门狗，根治方案（重试深度超时）待做。
6. **后台进程拉起方式**：交互 shell 里 `nohup ... &` 会随 shell 退出被杀，长期任务
   必须 `setsid nohup ... < /dev/null &`。

## 7. 复现入口

```bash
# 单题（续跑自动接 WM 状态）
KSEARCH_RUN_TAG=<tag> bash ksearch-run.sh <definition> 0 --wm
# 批量（题-卡映射见 scripts/ksearch_pool_campaign.sh TASK_GPU_MAP）
bash scripts/ksearch_pool_campaign.sh --pool <gpus> --concurrency <n> --tag <tag> --tasks <t1> <t2> ...
# 终评（FI）
scripts/ksearch_final_eval.py --task <def> --run-dir <run_dir> --trials 1
# 本归档构建
python scripts/build_h100_archive.py
```

环境/坑表详见 docs/H100_REPRODUCTION_GUIDE.md 与 docs/H100_RUNBOOK_v0.6.md。
