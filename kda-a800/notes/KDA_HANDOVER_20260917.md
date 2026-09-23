# KDA 任务池交接记录（2026-09-17）

> **[2026-09-18 注]** 本文为当时状态的历史记录。权威操作参考已更新为
> `notes/KDA_HANDOVER_20260918.md`（目录/脚本接口/配置语义/SOP/平台约束）；
> 终局结果见 `notes/reports/kda_30tasks_final_20260918.md`。

## 1. 当前结论

本次实验对 30 个题目执行 KDA 流程：`draft → plan → candidate`。每版 candidate 使用固定 5 个 feedback workload 做一次正确性和性能评测；自动 final 全量评测始终关闭。

截至 2026-09-17 最近一次全局审计：

| 指标 | 数量/状态 |
|---|---:|
| 总题目 | 30 |
| 至少有一个有效候选的题目 | 19 |
| 只有无效候选的题目 | 7 |
| candidate 评测登记次数 | 44 |
| 有效候选记录 | 31 |
| 无效候选记录 | 11 |
| 尚未完成写入的候选记录 | 2 |
| final 全量评测 | 0（按要求关闭） |
| 全局状态 | 26 token-stopped，2 searching，2 draft |

“有效候选”只表示通过固定 5 个 feedback workload，不代表通过全部 workload 的 final 评测。

## 2. 最近任务池状态

最近一轮 5-worker 重试池：

- Campaign：`/data1/workspace/weihongren/kda-control/campaigns/formal-kda-20260916-retry3-20260917.json`
- 日志：`/data1/workspace/weihongren/kda-control/campaigns/formal-kda-20260916-retry3-20260917.pool.log`
- 事件：`/data1/workspace/weihongren/kda-control/campaigns/formal-kda-20260916-retry3-20260917.events.jsonl`
- 启动时间：2026-09-17 19:57:39 +08:00
- 启动参数：5 workers，GPU 1–5，`max-turns=40`
- final evaluation：关闭

启动命令：

```bash
export KDA_ALLOWED_GPUS=1,2,3,4,5
python3 /data1/workspace/weihongren/kda-controller/run_kda_pool.py \
  --campaign /data1/workspace/weihongren/kda-control/campaigns/formal-kda-20260916-retry3-20260917.json \
  --workers 5 --max-turns 40
```

本次交接前已终止剩余 4 个长期 retry worker，避免继续消耗额度。任务池本身会在所有 future 返回后退出；恢复前必须先检查是否还有旧 pool 进程。

## 3. 最近 5 题状态

| 题目 | 最后阶段 | candidate | 当前观测 token/retry | 说明 |
|---|---|---:|---:|---|
| `gdn_prefill_qk4_v8_d128_k_last` | candidate | 1 | 1,098,308 / 10 | 已有有效 `c001`；新 candidate 未完成，连续 retry |
| `L2/040_altup_predict_correction_cycle_backward` | draft | 0 | 79,313 / 5 | draft 未完成，连续 retry |
| `L2/043_mamba_chunk_scan_with_segsum` | candidate | 0 | 470,145 / 5 | 首个 candidate 未完成，连续 retry |
| `L2/051_seqlen-finetuned-reconstructed_hyena_complete_forward_block` | draft | 0 | 281,390 / 20 | draft 未完成，历史 retry 严重 |
| `L2/080_moe_complete_layer_with_shared_expert_backward` | token-stopped | 1 | 1,041,373 / 0 | plan 和 `c001` stage 完成，达到 soft token limit |

`L2/080` 的最近 candidate stage：返回码 0，`artifact_ok=true`。其它 4 个 worker 在约 55 分钟内累计 8–10 次新 retry，最后停在 `api_retry`，没有形成新完整 artifact，因此已安全终止。

## 4. API retry 原因与处理标准

当前中转站问题主要有两类：

1. **HTTP 504**：修复代理将 Claude 流式请求转换为上游非流式请求；长 xhigh 回合接近 300 秒时，上游网关可能超时。
2. **HTTP 403**：曾发生额度或渠道不可用；之后探测恢复为 HTTP 200。

处理标准：

- 偶发 1 次 retry：继续观察。
- 同一阶段连续 3 次以上：标记关注。
- 连续 5 次以上，且 15–20 分钟无新的 assistant/tool 输出或 artifact：终止该阶段，fresh session 重跑。
- 失败 retry 不返回 usage，无法精确计算失败请求 token；必须保留 retry 数和 superseded transcript。
- 重跑前先用最多 8 token 的 API probe 确认 HTTP 200。

## 5. GPU 与可信评测

可信评测器：`/data1/workspace/weihongren/kda-controller/evaluate_candidate.py`。

已支持 GPU 白名单：

```bash
export KDA_ALLOWED_GPUS=1,2,3,4,5
```

评测器只会选择：

- 白名单内 GPU；
- 显存使用低于 200 MiB；
- utilization 为 0%；
- 未被 controller lock 占用。

Claude 不负责选择 GPU，只能调用：

```bash
./scripts/evaluate_candidate.sh feedback <candidate-id>
```

controller 负责 GPU 租约、空闲检查、计时和 foreign-process 监控。最近全局审计未发现可信评测期间的 foreign GPU process。

## 6. Claude 和候选约束

candidate prompt 当前强制：

- 每轮实现并评测恰好一个 immutable candidate；
- 只能调用固定 feedback 评测脚本；
- 禁止自动 final evaluation；
- 核心计算必须使用 Triton；
- 禁止 Torch、CPU、NumPy 或其它计算回退；
- 只允许 `Skill(KernelWiki)`；
- 禁止 `Skill(ncu-report-skill)`。

静态检查位于 `kda-controller/evaluate_candidate.py`。它要求存在 Triton JIT kernel，并拒绝常见 Torch 计算算子及 CPU/NumPy 回退。

曾经出现静态检查误判：docstring 中的 `torch.sort(...)` 被当成执行代码。现已用 Python `tokenize` 排除注释和字符串后再检查。修复后 `L1/058` 的 `c003` 已由可信脚本补测：`5/5` 正确，geomean `3.4109x`。

## 7. 重要目录和文件

- 工作区：`/data1/workspace/weihongren/kda-runs/`
- 控制状态：`/data1/workspace/weihongren/kda-control/`
- 控制器：`/data1/workspace/weihongren/kda-controller/`
- 原始 30 题 campaign：`kda-control/campaigns/formal-kda-20260916.json`
- 最近 5 题 campaign：`kda-control/campaigns/formal-kda-20260916-retry3-20260917.json`
- Claude 当前 transcript：各题控制目录的 `claude/`
- 历史中断 transcript：各题控制目录的 `claude-superseded-*`
- FlashInfer evaluator：`evaluators/evaluate.py`
- SOL evaluator：`evaluators/evaluate_sol.py`
- SSE 修复代理：`claude-opus-proxy/repair_proxy.py`
- 代理日志：`claude-opus-proxy/proxy.g0056.log`
- token 汇总器：`kda-observability/summarize_claude_transcript.py`

## 8. 恢复前检查

```bash
ps -ef | rg 'run_kda_pool|run_task_stage|claude -p'

source /data1/workspace/weihongren/llm.env
curl -sS -o /tmp/kda-probe.json -w '%{http_code}\n' \
  https://llmapi.isrc.ac.cn/v1/chat/completions \
  -H "Authorization: Bearer ${KDA_DEF_FEY:-$ISRC_API_KEY}" \
  -H 'Content-Type: application/json' \
  --max-time 30 \
  -d '{"model":"Claude-Opus-4.8","messages":[{"role":"user","content":"Reply only: OK"}],"max_tokens":8}'

nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits
```

API 必须返回 HTTP 200。选择真正空闲的 GPU；可使用 2–5 workers，但 Claude 并发越高，越容易放大中转站 300 秒超时问题。

## 9. 推荐恢复流程

1. 确认旧 pool、stage 和 Claude 进程均已退出。
2. 仅选择没有 `SEARCH_COMPLETE` 且没有 `TOKEN_LIMIT_REACHED` 的题目。
3. 对 retry 卡死题使用 fresh session：归档当前 `claude/*.jsonl`，删除 `.claude-session-id` 和 `POOL_BLOCKED`。
4. 不要删除 `state.json`、`candidates.jsonl`、candidate snapshots 或运行结果。
5. 生成新的 campaign，只包含待继续题目。
6. 根据空闲 GPU 设置 `KDA_ALLOWED_GPUS`，启动 2–5 workers。
7. 每 20 分钟检查：
   - stage 进程是否存在；
   - transcript 最后有效时间；
   - retry 次数及连续序列；
   - token 和输出长度；
   - `state.candidate_evaluations` 是否等于 `candidates.jsonl` 行数；
   - final evaluation 必须保持 0；
   - GPU monitor 是否发现 foreign process。
8. 单题异常时只终止该题 Claude 子进程，让 pool 处理返回；不要直接杀整个池。

## 10. 已知审计问题

- `L2/015_audio_sinusoidal_position_embedding_with_conv_projection`：`state.candidate_evaluations=1`，但 `candidates.jsonl` 为 0 行，存在账本不一致，后续总审计前必须修复或说明。
- 全局尚无 final 全 workload 评测，这是刻意关闭，不是遗漏。
- 目前 19/30 题至少有一个 feedback-valid candidate；不能据此宣称 19 题已通过官方完整 workload。
- 7 题目前只有无效候选：
  - `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`
  - `L1/002_vae_conv3x3_groupnorm_silu_residual_fused`
  - `L1/020_vision_patch_merger_spatial_shuffle_mlp`
  - `L2/012_moe_expert_batched_execution_with_capacity_factor`
  - `L2/015_audio_sinusoidal_position_embedding_with_conv_projection`
  - `L2/049_group_limited_topk_routing`
  - `L2/057_residual_coupling_flow_block`

## 11. 下一步优先级

1. 先确认 API 额度与 300 秒超时是否稳定，不要立即再次启动 5 个 xhigh worker。
2. 优先完成 `L2/040` 和 `L2/051` 的 draft；它们尚未进入候选阶段。
3. 完成 `L2/043` 的首个 candidate。
4. 对 `gdn_prefill`，先审阅已有有效 `c001` 和计划，再决定是否值得继续消耗 token。
5. 审计 `L2/080 c001` 的 feedback 结果和账本记录。
6. 修复 `L2/015` 的 state/ledger 不一致。
7. 所有题 terminal 后做全局一致性审计；没有用户明确许可时仍不运行 final evaluation。
