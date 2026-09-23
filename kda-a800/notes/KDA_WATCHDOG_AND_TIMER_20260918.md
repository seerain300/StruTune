# KDA 看护/对账/计时驱动 交接记录（2026-09-18 凌晨）

> 背景：retry4 池（4 题）运行期间，`gdn_prefill` 与 `L2/051` 因中转站 504 重试耗尽死亡（rc=1），
> pool 写 `POOL_BLOCKED` 后放弃，worker 空转近两小时。为此新增三层机制（经用户确认全部落地）。

## 1. 进程看护：`kda-controller/pool_watchdog.py`

- 只救 `POOL_BLOCKED` 且 `returncode=1` 的题（API 超时死亡）；rc=143（人工终止）永不碰。
- 救援前依次校验：API 探测 HTTP 200（经 curl——中转站对 Python urllib 的 TLS 指纹返回 503）、
  该题无存活 stage 进程、全局 `claude -p` 进程数 < 上限（默认 4）、该题救援次数 < 3（成功一次重置）。
- 救援动作：对账（见 §2）→ 删 `POOL_BLOCKED` → 起 `run_kda_pool.py --tasks <run_id> --workers 1`
  （自动 `--resume` 原会话）。
- 日志：`campaigns/<tag>.watchdog.jsonl`（动作）、`.watchdog-state.json`（计数）、
  `.rescue-<run_id>-<n>.log`。启停：`kill $(cat campaigns/<tag>.watchdog.pid)`。
- 当前实例：retry4 campaign，PID 见 `formal-kda-20260916-retry4-20260917.watchdog.pid`。

## 2. 账本对账：`kda-controller/reconcile_ledger.py`

不变式：`state.candidate_evaluations` == `candidates.jsonl` 可解析行数（交接审计规则）。

规则（`--apply` 时执行，否则报告）：
- A：`feedback_completed=true` 但无 `evaluation_index` 的候选 → 补计数（覆盖 v3 评测器的新窗口）。
- B：计数 > 账本行数 → 按缺失候选从 `runs/candidates/<id>/feedback.json` 自动补一行，
  带 `reconciled_by` 标记、decision 为 "operator-reconciled (pending review)"；只追加、永不改写旧行。
- C：账本行数 > 计数 → 仅报告（绝不自动减计数）。
- 空行跳过；非空坏行默认报告，`--repair-ledger` 可隔离到 `candidates.jsonl.fragments`（原文保留）。
- 账本 ID 字段兼容 `candidate_id` / `candidate` / `id` 三种历史写法。

2026-09-18 全量体检：29/30 题一致；`L2/015` 已按规则 B 修复（评测本身失败 valid=false，
账本补行如实记录）。`L1/058` 此前怀疑的"坏行"实为空行，无需修复；其 c003 可信补测
（5/5 正确，geomean 3.41×）仍是 ledger 中唯一有效候选记录，decision 仍是历史 reject，复核待做。

## 3. 评测器 v3 计数时机变更：`kda-controller/evaluate_candidate.py`

- 旧：feedback 评测**开始前**计数（interrupted 评测留下孤儿计数 → L2/015 类不一致的根源）。
- 新（v3）：计数移至评测**完成时**（或静态检查失败返回时），幂等标志 `feedback_counted`，
  `evaluation_index` 随计数赋值。中断的 benchmark 不再计数、不再占用评测预算。
- 审计注意：2026-09-18 00:40 前后的评测分别使用新旧口径；以 `feedback_counted` 标志区分
  （旧行记录无此标志）。预算检查语义不变（仍在评测前检查 counter 与 budget）。

## 4. 计时驱动：`kda-controller/timer_driver.py`（已就绪，未启动）

用途：对 token 耗尽题做"不限 token、外部计时"续跑。Claude 全程不知道时间限制。

每窗流程：对账 → 判定下一 stage → 等并发槽位 → 起 `run_task_stage`（独立进程组）→
等 `--window-minutes`（默认 40）：
- 到点且不在危险窗口（无存活评测进程 且 计数==账本）→ 立即 SIGTERM 进程组（安全点）。
- 在危险窗口 → 宽限轮询（30s 一次，上限 `--grace-minutes` 默认 20）：窗口关闭即杀；
  宽限耗尽 → SIGKILL，下一窗开跑前的对账兜底。
- 窗日志：`campaigns/<tag>.timer.jsonl`；每 stage 输出 `control/timer-<n>-<stage>.log`。

**启动 9 题（L1-002/020/058?、L2-012/015/036/049/057、dsa_sparse、gemm）前必须**：
1. 抬高各题 `task.json` 的 `token_soft/grace/hard_limit`（并同步改 TASK.md 中的限额文案）；
2. 删除各题 workspace 的 `TOKEN_LIMIT_REACHED`；
3. 组装新 campaign（沿用 retry3/4 手工组装方式）；
4. `KDA_ALLOWED_GPUS` 与 retry4 池错开（如 6,7），`--max-claude` 计入全局并发。

## 5. 与 SSE/504 问题的关系（决策：维持现状 + watchdog 兜底）

- 中转站流式 bug 实测仍在（2026-09-18 00:20 直测：工具参数后缺 `content_block_stop`/
  `message_delta`/`message_stop`/最终 usage）。
- 现行"非流式上游"保精确 usage 记账，代价是长回合撞网关 ~300s → 504（代理日志 107/1667）。
- 若未来切"流式+尾部修补"（代理已有 legacy 路径），需接受 output token 估算口径；
  建议另起端口实例灰度，不动现网。

## 6. token 记账口径提醒

- watchdog 救援不归档 transcript，死会话用量继续计入 budget（比人工归档口径更保守）。
- 504 失败请求不返回 usage，无法入账——真实消耗略高于账面，以 transport.api_retries 侧面观察。

## 9. 收尾记录（2026-09-18 10:10，经用户指令"每题一窗，无结果即停"）

- 10:07 停止全部机制：2 个 timer driver、watchdog、3 个在跑 stage（进程组 SIGTERM，全部干净退出）。
- 8 题写入 `OPERATOR_STOPPED` 标记（含窗口数/评测数/最优候选 JSON）；已完成 5 题保留原终态标记。
- `L2-057` 的 c004 账本行（评测被窗口击杀、从未产生结果）隔离至 `candidates.jsonl.fragments`，账本回到 4==4。
- 最终对账：30/30 全部通过不变式。
- 续跑阶段（09-17 21:22 → 09-18 10:08，约 12.8 小时）活跃会话累计 1.362 亿 budget token。
- 有结果 9 题 / 无结果 4 题（L2-015、L1-020、L2-036、L2-051），明细见终局报告。
- 重启任何一题：删 `OPERATOR_STOPPED` → `python3 kda-controller/timer_driver.py --campaign <campaign> --round-robin ...`。

## 8. 事故记录：评测器 v3 引入的 LOCK_EX bug（已修复）

- 2026-09-18 00:40 的 v3 计数改动中，两处 `fcntl.flock(lock, LOCK_EX)` 误写（缺 `fcntl.` 前缀，
  位于 count_feedback_evaluation 闭包和完成后记账块），导致 **00:40–01:22 之间所有 feedback
  评测在收尾记账一步 NameError 崩溃**：benchmark 照跑、feedback.json 已落盘，但
  `feedback_completed`/计数/评测结果返回给 Claude 全部缺失。
- 01:22 修复（两处改为 `fcntl.LOCK_EX`）。受影响的半完成记录（重跑同候选评测即自愈，
  feedback.json 已在盘仅作旁证）：
  - `gdn_prefill c002`（started 01:11）— rc=4 POOL_BLOCKED，01:24 手动清标记重启
  - `L2-012 c002`（started 01:14）— driver 下一 pass 自动重试
  - `dsa_sparse c002`（started 01:21）— 窗口内自动重试
- `L1-058` state 中 c001/c002 的 started-but-not-completed 是 09-17 凌晨 quota 403 时代的
  旧残留，与本事故无关（该题 terminal，未纳入续跑）。
- 教训：评测器改动后应先用 `--help` 或最小 dry-run 实测一次，而不是只 py_compile。

## 7. 过夜配置快照（2026-09-18 01:10 起，经用户确认）

- **retry4 池**（GPU 1–5 评测）：4 题，watchdog（`--max-claude 6`）自动救 rc=1 死亡。
- **timer 续跑**（GPU 6/7 评测）：9 题已解除 token 限制（v2 版本化配置）、fresh session、
  标记清除，分两组 round-robin：
  - `timer-continuation-20260918-a`：L2-012 / L2-015 / L2-049 / L2-057
  - `timer-continuation-20260918-b`：dsa_sparse / gemm / L1-002 / L1-020 / L2-036
  - 参数：60 分钟窗口 + 20 分钟宽限 + 安全点击杀，每题每 pass 一窗，最多 8 pass，
    全局 claude 并发上限 6（与 retry4 共享）。
  - 准备留档：`timer-continuation-20260918.preparation.json`（历史 token/retry）+
    `timer-continuation-20260918.config-versions.json`（限额版本化记录）。
- **早晨检查顺序**：
  1. `campaigns/timer-continuation-20260918-{a,b}.driver.log` 尾部 + `*.timer.jsonl`（每窗 outcome）
  2. `campaigns/formal-kda-20260916-retry4-20260917.events.jsonl`（retry4 是否收尾）
  3. `python3 kda-controller/reconcile_ledger.py --campaign kda-control/campaigns/formal-kda-20260916.json`（全量对账）
  4. 各题 `candidates.jsonl` 新增行（timer 续跑的有效候选会在此出现）
  5. GPU：`nvidia-smi`（6/7 应只在评测时段被占）
- 进程清单：pool 4161594 / watchdog `.watchdog.pid` / driver `.driver.pid` ×2（均在 campaigns/ 目录）。
