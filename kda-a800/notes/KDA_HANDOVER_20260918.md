# KDA 实验运行手册（2026-09-18 版）

> 本文是权威操作参考：目录结构、脚本接口、配置语义、标准操作流程（SOP）。
> 历史与过程记录见 `KDA_HANDOVER_20260917.md`（上一任交接）、`KDA_WATCHDOG_AND_TIMER_20260918.md`（续跑机制与事故）、
> `notes/reports/kda_30tasks_final_20260918.md`（30 题终局结果）。

## 1. 目录结构

```text
/data1/workspace/weihongren/
├── kda-runs/<run_id>/              # 每题独立工作区（Claude 只能在此活动）
│   ├── CLAUDE.md / TASK.md          # 任务合同（工作流规则、停止条件、预算声明）
│   ├── task/definition.json         # 官方题目定义
│   ├── task/feedback_workloads.jsonl# 反馈 workload 集（默认全量，粗测口径）
│   ├── docs/{draft,plan}.md         # 阶段产物
│   ├── solution/solution.py         # 当前候选源码
│   ├── runs/candidates/<cXXX>/      # 每候选评测产物：feedback.json / final.json / *.log
│   ├── candidates.jsonl             # 证据账本（每评测一行，只追加不改写）
│   ├── final/<cXXX>/                # 旧版 final 结果路径（evaluation/performance.json）
│   └── scripts/evaluate_candidate.sh# Claude 唯一可调的评测入口
├── kda-control/<run_id>/            # 控制侧（Claude 禁止访问）
│   ├── task.json                    # 可信任务配置（限额、预算、评测参数）
│   ├── state.json                   # 计数器与候选记录（评测器维护）
│   ├── candidates/<cXXX>/solution.py# 候选不可变快照（sha256 校验）
│   ├── claude/*.jsonl               # 活跃会话 transcript（0001-<stage>.jsonl 编号）
│   ├── claude-superseded-<原因>-<日期>/  # 归档会话（token 历史留痕）
│   ├── observability.json           # token/retry 汇总（summarizer 维护）
│   └── reconciliation.jsonl         # 账本修复留痕
├── kda-controller/                 # 全部控制脚本（见 §2）
├── kda-observability/summarize_claude_transcript.py
├── evaluators/{evaluate.py,evaluate_sol.py}   # FlashInfer / SOL 官方评测器
├── claude-opus-proxy/              # SSE 修复代理（127.0.0.1:18901）
├── bin/claude-kda-opus48           # Claude 启动包装（自动起代理、加载 llm.env）
├── llm.env                         # 密钥（KDA_DEF_FEY 优先，权限 600）
└── notes/, kda-control/campaigns/  # 文档与 campaign 清单
```

## 2. 控制脚本接口（kda-controller/）

### 2.1 run_kda_pool.py — 任务池调度
```bash
python3 kda-controller/run_kda_pool.py --campaign <campaign.json> \
  [--workers N(1-6)] [--tasks 'fnmatch模式'] [--max-turns 40] [--dry-run]
```
- 每题循环判定下一阶段 `draft → plan → candidate`，直到终态标记出现；
- 阶段失败（rc≠0）写 `<workspace>/POOL_BLOCKED` 并放弃该题（由 watchdog 回收）；
- `--dry-run` 只打印每题下一阶段；final 评测永远不自动执行。

### 2.2 run_task_stage.py — 单阶段执行（pool 与 watchdog 的执行单元）
```bash
python3 kda-controller/run_task_stage.py <draft|plan|candidate> \
  --workspace <kda-runs/<run_id>> [--max-turns 40] [--force]
```
- 会话规则：workspace 存在 `.claude-session-id` 则 `--resume`，否则新会话并写入该文件；
- token 闸门（stage 启动前读 task.json）：`soft` 拒绝新 candidate、`grace`/`hard` 全停并写 `TOKEN_LIMIT_REACHED`；
- 产物校验：draft/plan 要求文件非空；candidate 要求 `state.candidate_evaluations` 恰好 +1（或已写 `SEARCH_COMPLETE`）；
- 返回码：0 成功；claude 自身 rc 原样透传；claude 正常退出但产物不符 → 4。

### 2.3 evaluate_candidate.py — 可信评测器（核心）
```bash
# Claude 侧唯一入口（workspace 内）：
./scripts/evaluate_candidate.sh feedback <cXXX>
# 运营者直调：
python3 kda-controller/evaluate_candidate.py final --candidate <cXXX> --workspace <ws> \
  [--gpu-wait-timeout 300]
```
- **GPU 租约**：仅选 `KDA_ALLOWED_GPUS` 白名单内、显存 <200MiB、利用率 0%、未被 flock 锁定的卡；
- **静态检查**：必须有 `@triton.jit` 核；禁止 `torch.compile/.cpu()/.numpy()/numpy/异常回退` 及
  `torch.(matmul|mm|bmm|einsum|sum|mean|rsqrt|sqrt|pow|norm|softmax|topk|sort|conv*|linear)` 前缀形式
  （**注意**：`.sort(` 等方法形式与 `bincount/cumsum/repeat_interleave` 尚未覆盖，需人工审计）；
- **不可变候选**：`cXXX` 与 source sha256 绑定，改动源码必须换新 ID；快照冲突直接拒绝；
- **计数语义（v3）**：`candidate_evaluations` 在评测**完成时**才 +1（幂等标志 `feedback_counted`），
  中断的评测不计数、不占预算；静态检查失败同样计数并记录；
- **final 语义**：`--stage final` 全量 workload、`reference_cache=false`（参考实现同进程成对实时计时）；
  运行期 foreign-process 监控发现外部 GPU 进程即判废（rc=3，timing invalidated）；
- 产物：`runs/candidates/<cXXX>/{feedback,final}.json`。

### 2.4 pool_watchdog.py — 失败自动回收
```bash
python3 kda-controller/pool_watchdog.py --campaign <campaign.json> \
  [--max-claude 6] [--max-rescues 3] [--interval 60]
```
- 只回收 `POOL_BLOCKED` 且 returncode ∈ {1(API 死亡), 4(产物校验失败/多评)} 的题；rc=143（人工终止）永不碰；
- 回收前自动对账（reconcile_ledger）→ 删标记 → 起单题 pool（自动 resume）；
- 回收前置条件：API 探测 HTTP 200（经 curl——中转站对 Python urllib 返回 503）、无存活 stage、全局 claude 数低于上限、该题救援次数未满（成功一次重置）；
- 状态/日志：`campaigns/<tag>.watchdog-state.json` / `.watchdog.jsonl`；
- **注意**：外部修改 state 文件会被其内存副本覆盖，改后须重启 watchdog。

### 2.5 reconcile_ledger.py — 账本对账
```bash
python3 kda-controller/reconcile_ledger.py --campaign <campaign.json> | --run <run_id> \
  [--apply] [--repair-ledger]
```
- 不变式：`state.candidate_evaluations` == `candidates.jsonl` 可解析行数；
- 规则 A：completed 但无 `evaluation_index` → 补计数；
- 规则 A2：started 未完成但 `feedback.json` 含完整 per-workload 结果（中断受害者）→ 补计数并封闭记录；
- 规则 B：计数 > 账本 → 从 feedback.json 自动补写带 `reconciled_by` 标记的证据行（只追加）；
- 规则 C：账本 > 计数 → 若多余行的候选有未完成 state 记录则放行自对齐（重评后追平），无 state 记录才报 needs_human；
- 账本 ID 字段兼容 `candidate_id` / `candidate` / `id` 三种历史写法；
- `--repair-ledger` 把不可解析行隔离到 `candidates.jsonl.fragments`（原文保留）。

### 2.6 timer_driver.py — 限时续跑（外部计时，Claude 不知情）
```bash
export KDA_ALLOWED_GPUS=<白名单>
python3 kda-controller/timer_driver.py --campaign <campaign.json> \
  --round-robin --max-passes 8 --window-minutes 60 --grace-minutes 20 \
  --max-turns 40 --max-claude 6 [--dry-run]
```
- 每题每 pass 一窗；到窗线检查危险窗口（该题评测进程存活 或 计数≠账本）：
  不在 → 立即 SIGTERM 进程组（安全点）；在 → 最多宽限 N 分钟，窗口闭合即杀，超时 SIGKILL；
- 每窗开启前自动对账；needs_human 的题跳过并记录；
- 窗日志 `campaigns/<tag>.timer.jsonl`，每窗 stdout 落 `kda-control/<run>/timer-<N>-<stage>.log`；
- `--dry-run` 列出每题下一阶段并预警 token 限额不足的题。

### 2.7 辅助
- `campaign_status.py --campaign <json>`：池级进度汇总；
- `prepare_batch.py`：从 experiment_manifest.json 全新建题（resume 续跑不用它）；
- `kda-observability/summarize_claude_transcript.py <transcripts...> --output x.json`：
  token 汇总（usage 记录去重、以会话末尾完整 result 记录为准；budget = 未缓存输入+缓存写+缓存读+输出）。

## 3. 配置语义

### 3.1 campaign JSON（`kda-control/campaigns/<tag>.json`）
```json
{ "schema": "kda-campaign-v1", "tag": "...", "manifest": "...", "manifest_sha256": "...",
  "task_count": N, "workload_count": 738,
  "tasks": [ {"run_id","benchmark","task","workspace","control","workload_count","feedback_indices","feedback_uuids"} ] }
```
续跑 campaign 为手工组装：从旧 campaign 拷贝 task 条目、换 tag。池/watchdog/driver 均按此文件圈定范围。

### 3.2 task.json 关键字段
| 字段 | 语义 |
|---|---|
| `token_soft/grace/hard_limit` | 三档 token 闸门（原始 1M/1.5M/1.65M；v2 续跑统一 50M 兜底，见 §4.4） |
| `candidate_evaluation_budget` / `final_evaluation_budget` | 评测次数预算（final 默认 1，被外部污染判废后可版本化上调） |
| `evaluation.feedback/final` | warmup/iterations/seed/timeout；`reference_cache` 仅 feedback 可为 true |
| `token_limit_revision` / `final_budget_note` | 版本化修改标记（原值存 `task.token-v1.json` / `task.final-budget-v1.json`） |

### 3.3 环境变量
| 变量 | 语义 |
|---|---|
| `KDA_ALLOWED_GPUS` | 评测器 GPU 白名单（逗号分隔；**必须排除被外部占用的卡**） |
| `KDA_DEF_FEY`（llm.env） | KDA 专用 key（优先于 KDA_KEY/ISRC_API_KEY/LLM_API_KEY/NEW_API_KEY） |
| `ANTHROPIC_BASE_URL` | 由 `bin/claude-kda-opus48` 自动指向 `http://127.0.0.1:18901`（本地 SSE 修复代理） |

### 3.4 标记文件（workspace 根）
| 标记 | 写入者 | 语义 |
|---|---|---|
| `SEARCH_COMPLETE` | Claude | 自主收敛，含原因 |
| `TOKEN_LIMIT_REACHED` | run_task_stage | token 闸门收尾（正常终态） |
| `POOL_BLOCKED` | pool | 阶段 rc≠0；内容含 returncode（watchdog 按 1/4 回收，143 不碰） |
| `OPERATOR_STOPPED` | 运营者 | 人工停止；JSON 含窗口数/评测数/最优候选 |
| `.claude-session-id` | run_task_stage | 存在即 resume；删除即新会话 |

### 3.5 state.json
`candidate_evaluations` / `final_evaluations` 计数器；`candidates[cXXX]` 含 source_sha256、
`feedback_started_at/completed_at/returncode`、`feedback_counted`、`evaluation_index`。
修改状态文件后需重启依赖它的常驻进程（watchdog）。

## 4. 标准操作流程（SOP）

### 4.1 启动前检查
```bash
ps -ef | grep -E 'run_kda_pool|run_task_stage|claude -p' | grep -v grep     # 无残留
source /data1/workspace/weihongren/llm.env
curl -sS -o /dev/null -w '%{http_code}\n' https://llmapi.isrc.ac.cn/v1/chat/completions \
  -H "Authorization: Bearer ${KDA_DEF_FEY:-$ISRC_API_KEY}" -H 'Content-Type: application/json' \
  --max-time 30 -d '{"model":"Claude-Opus-4.8","messages":[{"role":"user","content":"OK"}],"max_tokens":8}'
# 必须 200；探测用 curl（Python urllib 会被中转站 503）
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
```

### 4.2 常规池（token 限额内）
```bash
export KDA_ALLOWED_GPUS=<空闲卡>
nohup python3 kda-controller/run_kda_pool.py --campaign <campaign.json> \
  --workers 4 --max-turns 40 > kda-control/campaigns/<tag>.pool.log 2>&1 &
echo $! > kda-control/campaigns/<tag>.pool.pid
# 建议同时起 watchdog（§2.4）自动回收 rc=1/4
```

### 4.3 限时续跑（不限 token，外部计时）
1. 对每题做 fresh-session 准备：`claude/*.jsonl` 移入 `claude-superseded-<原因>-<日期>/`，
   删 `.claude-session-id` 与 `POOL_BLOCKED`，历史量写入 `<tag>.preparation.json` 留档；
2. 版本化解除限额（§4.4），同步改 TASK.md/CLAUDE.md 的预算文案（否则模型按旧合同自我节流）；
3. 删 `TOKEN_LIMIT_REACHED`，组装新 campaign；
4. 启动 driver（建议 2 个实例分组、GPU 与其他池错开）+ watchdog；`--dry-run` 先验证阶段判定。

### 4.4 版本化修改限额（模板）
改 `task.json` 前把原文件备份为 `task.token-v1.json`；新值加 `token_limit_revision` 与说明字段。
TASK.md 中"Token soft limit…"句与 CLAUDE.md 第 8 条停止规则同步替换为外部管理声明。

### 4.5 全量 final 评测（运营者专属）
```bash
# 1) 恢复目标候选源码（评测器校验 solution.py 与记录哈希一致）：
cp kda-control/<run>/candidates/<cXXX>/solution.py  kda-runs/<run>/solution/solution.py
# 2) 独占空卡上执行：
KDA_ALLOWED_GPUS=<完全空闲卡> python3 kda-controller/evaluate_candidate.py final \
  --candidate <cXXX> --workspace kda-runs/<run> --gpu-wait-timeout 300
```
- **必须独占空卡**：运行期任何外部 GPU 进程都会导致 timing invalidated（rc=3）；
- 被判废的尝试会消耗 `final_evaluation_budget`，重试前按 §4.4 版本化上调；
- 结果在 `runs/candidates/<cXXX>/final.json`；旧批次结果在 `final/<cXXX>/{evaluation,performance}.json`。

### 4.6 收尾与重启
- 人工停止：杀 pool/driver 的**进程组**（driver 被杀不会终止其子 stage，需一并处理）；
  被停题写 `OPERATOR_STOPPED`（含原因与最优候选 JSON）；
- 重启任何被停题：删对应标记 → pool（`--tasks <run_id> --workers 1`）或 timer_driver 指定该题；
- 全量对账收尾：`reconcile_ledger.py --campaign ... `（30/30 OK 为通过）。

### 4.7 反馈评测口径（v2 默认：全量粗测）

自 2026-09-18 起，新建任务（`prepare_batch.py`）与续跑题的反馈评测默认为**全量 workload 粗测**：

| 维度 | 反馈（feedback） | 全量终测（final） |
|---|---|---|
| workload 覆盖 | **全部**（含边界形状） | 全部 |
| 迭代精度 | **粗测：warmup 2 / 10 iterations** | **精测：warmup 3–10 / 100 iterations** |
| 参考实现计时 | 缓存（SOL，`reference_cache=true`） | 不缓存，同进程成对实时计时 |
| 用途 | 搜索内循环：正确性全形状覆盖 + 粗略速度排序 | 权威论文数字 |

- 精度差异的语义：反馈的 speedup 是粗测排序信号（10 次迭代噪声约 ±5–10%），
  候选间比较与最终声明一律以 final 为准；
- 评测超时随 workload 数自动放大（`timeout_per_workload_seconds × count`）；
- 历史审计注意：2026-09-16 campaign 的原始协议为"固定 5 抽样 / 100 iterations"反馈，
  该批次及 09-18 前的候选记录均在此口径下产生（详见终局报告脚注）；
  `L1-005` 于 09-18 中途切换为"全量 / 100 iterations"（v1 备份 `*.feedback-v1.*` 可考）；
- 将存量题切换到本口径的版本化步骤：备份双侧 `feedback_workloads.jsonl` 与 `task.json` →
  写入全量集 → 更新 `feedback_workload_sha256 / feedback_indices=[0..N) / feedback_uuids=全部 /
  evaluation.feedback.{warmup:2, iterations:10}` → TASK.md 文案同步（备份 v1）。

## 5. 平台特性与既知约束（结论式）

1. **中转站流式截断**：上游 SSE 缺失结束事件 → 本地代理转非流式并重建 SSE（usage 精确）。
   代价：单回合生成超 ~300s 即被网关掐断（504）；长文档类单回合任务（如大 draft）通过率低。
2. **prompt 缓存近乎失效**（cache_write≈0、命中 2–16%）：resume 长会话的 token 按
   轮数×全量上下文重复计费（实测放大 10–46 倍）。**高轮数续跑优先 fresh session**；
   token 账面 ≠ 实际新增信息量。
3. **失败请求（504）不返回 usage、不入账**；真实消耗略高于账面。retry 计数见 observability.transport。
4. **静态检查方法形式盲区**：`x.sort(/.cumsum(/torch.bincount/repeat_interleave` 不在禁止清单——
   论文口径候选需按 `notes/reports/kda_30tasks_final_20260918.md` §4 的审计方法人工复扫。
5. **共享机器**：GPU 会被其他用户周期性占用/探测；final 计时必须独占空卡并接受判废重试。

## 6. 当前状态快照（2026-09-18 13:00）

- 30 题全部终态；全量 final 已跑 26 题（**20 题 >1× 通过**、6 题 <1× 通过、4 题无候选、
  INVALID 已清零）；`L2/012 c006` 为唯一 torch 回退（<1×，无指标损失）。
- 全部进程已停；campaigns 目录含 pid 文件的均为历史实例。
- 下一轮若继续优化：建议 §4.3 + 每候选 fresh session + 首个有效候选前预算上限 ~2000 万 +
  连续无效候选早停（阈值 ≥10，参考 L2/049 第 10 个候选才破零的实测）。

## 7. 新机器部署清单与检查事项

> 前提：新机器已装好 原始题目集 / 测评环境 / KDA 基础仓库。
> 打包件 `kda_ops_bundle_20260918.tar.gz` 补齐其余运维栈（脚本、代理、文档、campaign 模板）。

### 7.1 压缩包内容
```text
kda-controller/        # 全部 8 个控制脚本（含 v3 评测器、watchdog、对账、timer_driver、v2 prepare_batch）
claude-opus-proxy/     # repair_proxy.py + start.sh（SSE 修复代理）
kda-observability/     # summarize_claude_transcript.py
evaluators/            # evaluate.py / evaluate_sol.py（与 task.json 的 sha256 锁定配套）
bin/                   # claude-kda-opus48 / kda-eval / claude-kda / claude-shared
activate-claude.sh
experiment_manifest.json   # prepare_batch 的输入清单
campaigns/              # campaign JSON 模板 + 各 preparation.json 审计留档
notes/                  # 本手册 + 终局报告 + 过程档案
DEPLOY.md               # 本节独立版
```
**不含**（需手动处理）：`llm.env`（密钥，手动拷贝并 `chmod 600`）、历史任务工作区与 transcript
（留在原机器）、GPU 缓存。

### 7.2 路径适配（解包后逐项确认）
| 位置 | 旧机器值 | 检查/改法 |
|---|---|---|
| 全部 `kda-controller/*.py` 的 `ROOT` | `/data1/workspace/weihongren` | 改为新机器工作区绝对路径（8 个文件同改） |
| `evaluate_candidate.py` / `prepare_batch.py` 的 `SOL_ROOT`/`SOL_DATASET` | `/data1/workspace/ziming/dataset/SOL-ExecBench/...` | 改为新机器 SOL 题集位置 |
| `evaluate_candidate.py` 的 `FLASHINFER_DATASET` | `$ROOT/dataset/flashinfer-test` | 确认新机器 flashinfer 题集位置 |
| `SOL_PYTHON` / `SOL_CLI` | SOL 题集自带 `.venv/bin/python` | 确认新机器该 venv 存在且可跑 `evaluate_sol.py` |
| `FLASHINFER_PYTHON` | `/data1/workspace/ziming/miniconda3/envs/mtmc/bin/python` | **别人的 conda 环境**——新机器换成自己的（需含 torch/triton） |
| `CUDA_HOME` | `/usr/local/cuda-12.4` | 按新机器 CUDA 版本改 |
| `bin/claude-kda-opus48` 内绝对路径 | weihongren 家目录 | 按新机器改 |
| `prepare_batch.py` 的 `SOL_DATASET` | 同上 ziming 路径 | 同步改 |

### 7.3 环境检查（按序）
1. **Python**：控制器仅用标准库（py3.8+）；评测 python 见上表；
2. **Claude Code**：`~/.local/bin/claude` 存在；`CLAUDE_CONFIG_DIR` 指向共享配置（`.claude-kda`）；
   skills：`KernelWiki`（`ncu-report-skill` 被禁用可不装）；
3. **密钥**：`llm.env` 手动到位（`KDA_DEF_FEY` 优先链见 §3.3），`chmod 600`；
4. **代理**：首次运行任意 `bin/claude-*` 会自动拉起 127.0.0.1:18901；健康检查
   `curl --noproxy '*' http://127.0.0.1:18901/health`；
5. **API**：§4.1 的 curl 探测必须 200（**不要用 Python urllib 探测**，中转站对其返回 503）；
6. **GPU**：`nvidia-smi` 确认空闲卡；`KDA_ALLOWED_GPUS` 只放全空卡（<200MiB 且 0% 才能被租约）。

### 7.4 冒烟测试（新机器首跑顺序）
```bash
# 1) 静态：阶段判定与配置读取
python3 kda-controller/run_kda_pool.py --campaign <campaign.json> --dry-run
python3 kda-controller/timer_driver.py --campaign <campaign.json> --dry-run
# 2) 代理与模型链路（workspace 内）：
./scripts/evaluate_candidate.sh feedback <test-candidate>   # 一次反馈评测（租约/静态检查/计数链路）
# 3) 对账与看护：
python3 kda-controller/reconcile_ledger.py --campaign <campaign.json>
# 4) 观察一个 40 分钟窗口的完整生命周期后，再放开并发
```

### 7.5 历史审计口径（引用需知）
2026-09-16 campaign 的反馈协议为"固定 5 抽样 / 100 iterations"；09-18 起新建题默认
"全量 / warmup2 / iters10 粗测"（§4.7）。两口径的候选记录不可直接混比，
结果引用见 `notes/reports/kda_30tasks_final_20260918.md` 脚注。
