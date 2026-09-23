# K-Search H100 复现 RUNBOOK v0.6（2026-09-20，覆盖 REPRODUCTION_GUIDE 的差异部分）

> 本文件记录相对 `k-search_H100_REPRODUCTION_GUIDE.md`（终版）的全部协议与代码变更。
> 指南其余部分（路径适配、key 链路、-inf 补丁、报告/归档流程）仍有效。

## 1. 评测协议变更（反馈 vs 最终）

| 维度 | 反馈测评（搜索中，每轮候选） | 最终精确测评（阶段3） |
|---|---|---|
| workload | **全量**（`--num-feedback-workloads` 默认 1000000=全部；旧协议抽 5） | 全量（不变） |
| iterations | **默认 20**；慢 ref 题降为 **10**（下表） | 100（不变） |
| warmup / trials | 不变：FlashInfer 3/5，SOL 10/1 | 不变：FlashInfer 3/1，SOL 官方默认 |

慢 ref 题（反馈 10 iter，其余 20 iter），在 `ksearch-run.sh` 的 case 表维护：
- FlashInfer：`gdn_prefill_qk4_v8_d128_k_last`、`mla_paged_prefill_causal_h16_ckv512_kpe64_ps1`、`gqa_paged_prefill_causal_h32_kv8_d128_ps1`、`gqa_paged_decode_h32_kv8_d128_ps1`
- SOL：`L2/036_convnextv2_layer_with_nhwc_persistence_backward`

ref 延迟有磁盘缓存（首轮计时代价高、之后只计时候选）。

**Prompt 已加 workload 截断描述**（`_feedback_workloads_text`，两种任务同款）：
在 definition 文本末尾追加 "Feedback Workloads (N total, all are evaluated)" 段——
展开前 8 个 workload 的 axes 取值，其余按每个 axis 的 min..max 范围汇总
（例：gdn_prefill 100 wl → 8 行 + "92 more ... axis ranges: total_seq_len 35..8192, ..."）。
全量信息另经每轮反馈进入（passed/total、延迟汇总、失败 workload 日志）。

## 2. GPU 使用变更：30 题全部动态卡池

**卡池范围（2026-09-20 决定）：只用 GPU 0-2**（`--pool "0,1,2"`）。
gpu3-5 有跨容器外部租户且其进程在 compute-apps 查询里不可见（只有 memory.used
能暴露），不要进池；0-2 是我们持有器控制的卡。

- FlashInfer 任务已接入 `KSEARCH_GPU_POOL`（`flashinfer_bench_task.py` 的两处
  `Benchmark.run_all` 包 `_pool_gpu_env`；baseline/seed eval 复用同一包装）。
- **阶段 2a（FlashInfer 10 题）改用 pool_campaign**（不再用 ksearch_campaign.sh 一卡绑一题）：
  ```bash
  KSEARCH_STRICT_NO_LIB=1 KSEARCH_WM_MAX_ACTION_NODES=20 KSEARCH_WM_MAX_ATTEMPTS_PER_NODE=5 \
  bash scripts/ksearch_pool_campaign.sh --pool "0,1,2" --concurrency 5 --tag formal_h100 --tasks \
      dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 gdn_decode_qk4_v8_d128_k_last \
      gdn_prefill_qk4_v8_d128_k_last gemm_n4096_k4096 gqa_paged_decode_h32_kv8_d128_ps1 \
      gqa_paged_prefill_causal_h32_kv8_d128_ps1 gqa_ragged_prefill_causal_h32_kv8_d128 \
      mla_paged_decode_h16_ckv512_kpe64_ps1 mla_paged_prefill_causal_h16_ckv512_kpe64_ps1 rmsnorm_h4096
  ```
  阶段 2b/2c（SOL）命令不变（本来就是池化）。
- `pool_campaign.sh` 的 run_dir 已支持两种任务（L1/L2 前缀 → sol 目录，否则 ksearch 目录）。

## 3. gpu_pool 与占卡管理器联动（自占卡入池）

`k_search/utils/gpu_pool.py` holder-aware（默认开，`KSEARCH_POOL_HOLDER_AWARE=0` 关闭）：
- 自家 fill_vram 持有器（`/tmp/mtmc-gpu-occupancy/gpu-N.pid` 存活且 cmdline 含 fill_vram）
  守着的卡视为**可用**，不再是"租户占用"；
- 领到池锁后：持有器在跑 → `gpu_occupancy.py stop` 让位 → 复核无陌生租户 → benchmark；
- benchmark 结束/异常退出 → 自动 `start` 占回（与 evaluate.py 的 lease 语义一致）；
- 持有器之外仍有陌生进程的卡照旧踢出池。

效果：搜索全程（含 LLM 长空闲）持有器常驻守卡，仅计时窗口显存腾空。
**LLM 间隙也占卡**：反馈 benchmark 结束后无条件 `ensure_started`（`KSEARCH_POOL_CLAIM_EMPTY=1`
默认开）——原本就有持有器的卡恢复原持有器；真空卡则转为我们的持有器常驻，
即"我们在占"的显式状态。campaign 全部结束后，对**我们 claim 的（原本空的）卡**
按需 `gpu_occupancy.py stop --gpu N` 恢复原状；原有持有器的卡保持 running。

**租户识别 v2（进程身份制，替代显存阈值）**：`classify_gpu_processes` 对卡上每个 GPU 进程分类——
holder（cmdline 含 fill_vram，只经管理器停/启）/ own（进程树命中 或 environ 带 KSEARCH_OWNER_TAG，
ksearch-run.sh 每次 run 自动生成并传给全部子进程）/ stranger（其余一切，含 /proc 不可见的跨容器进程）。
只有 stranger 非空才判"有租户"并跳过该卡、**不动持有器**。自家主进程的常驻 CUDA 上下文
（~900MB）不再误判（v1 显存阈值判空曾导致领卡循环每 30s 停/启持有器自锁，2026-09-20 冒烟事故）。
同账号同容器里同事跑的同名脚本因无我们的 tag 且不在我们进程树里 → 正确归为 stranger。

**防冲突规则（开跑前必做）**：
1. `gpu_occupancy.py status --gpu N` + `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv`
   逐卡检查；持有器 pid 在短期内反复变化（活跃 lease 循环特征）= 有同事正在用，**该卡不进 --pool**；
2. `--pool` 只放：真空闲卡 + 我们持有器静态守着（pid 稳定、无 lease 活动）的卡；
3. 池的 holder-aware 联动只在 benchmark 计时窗口停持有器，不影响其他使用者。

## 4. 续跑语义（不变，对两种任务均有效）

pool_campaign 无 DONE 时：数 `campaign_stdout.log` 里 "Optimization Round" 行数 →
剩余预算 = 100 − 已做（下限 5）→ `--continue-from-world-model auto` 从
`ksearch-artifacts/<题名>/world_model/world_model.json` 恢复。

## 5. 反馈全量化的代价预估（首轮，ref 未缓存时）

20 iter 题：ref 全量计时一次性付清后入缓存；10 iter 题按用户实测表
（036_convnextv2 ~24min@100iter → ~5min@10iter首轮、后续轮仅候选计时）。

## 6. 运行期修复记录（2026-09-20 晚，v8→v12）

1. **入口参数补齐**：--seed/--skip-final-eval/--wm-max-action-nodes/--wm-max-attempts-per-node
   （源机器未提交修改；节点上限已在生成器 v2 循环实现，20 节点×5 尝试）
2. **_gpu_empty v3（不可见余量法）**：总占用 − Σ可见进程占用 ≤1000MB 才判空。
   v2 的"总占用≤2000MB"曾致单上下文+worker>2GB 全池死锁（17:40 事故）
3. **全局公共队列**：pool.queue.lock 内核 FIFO 排队 + 出队时挑空闲卡。
   取代各任务独立轮询（手气制）与蹲守单卡（v4 缺陷：卡分配偏斜）
4. **延迟回占（REHOLD_DELAY=45s）**：连续窗口场景不再每窗口白爬 71GB 持有器；
   空闲确认后由后台 watcher 占回
5. **守卫 v2**：删除白名单句（保留 BANNED 清单 + tl.dot + REJECTED）
6. **每轮代码落盘**：ksearch-artifacts/<题>/rounds/round_NNN_<status>.py（可审计；
   已验证 gdn_decode 413x 为真 Triton：@triton.jit×1、零 banned 调用）
7. **父进程可见性**：CUDA_DEVICE_ORDER=PCI_BUS_ID + 池模式下父进程默认只见池首卡
   （修复 6 卡上下文泄漏）；campaign 共享 KSEARCH_OWNER_TAG（修复跨任务误判饿死）
8. **卡池扩容**：18:20 起 --pool "0,1,2" --concurrency 10（FlashInfer 全 10 题）

## 7. 终版架构（v15，2026-09-20 18:45，第六次故障后定稿）

**任务-卡终身绑定（sticky affinity）**：campaign 按任务序号轮转分卡（4/3/3），
KSEARCH_TASK_GPU 出生即钉死 CUDA_VISIBLE_DEVICES，父/子进程只可能碰自己的卡。
**删除延迟回占 watcher**，恢复卡锁内同步启停持有器（持有器与 benchmark 的互斥
由锁硬保证，异步 watcher 曾与领卡竞态导致两者共存砸卡）。保留：全局 FIFO 队列、
进程身份判空（不可见余量法）、窗口后 empty_cache 归还死张量。

六次故障共同根因：父进程直接做 GPU 工作的任务在卡间漂移（阈值误判→死锁→
偏斜→错位→累计→竞态皆为漂移的补丁连锁）。绑定后漂移在结构上不存在。

## 8. 收官期事故补记（2026-09-23，FI 三题续跑期间）

1. **假 DONE 复发**：gqa_paged_prefill 以 rc=0 退出并写 DONE，实际 96/100 轮
   （RESUME 给的 9 轮预算中 4 次 WM refine 重试未产出评测）。处置：删 DONE 标记
   单题补跑至 106。规则不变：**DONE ≠ 100 轮，重启前必须 grep -c "Round summary"
   核实**。附带发现 floor-5 规则副作用：done_evals=100 时补拉又给 5 轮
   （mla_paged_prefill 被拉到 105 轮，手动停+标 DONE），预算=top-up 时应设 0。
2. **双 campaign 互斥死锁**：同一张卡上的任务被两个 campaign 实例分别拉起时，
   二者的 KSEARCH_OWNER_TAG 不同，污染检测器互判对方为陌生租户，评测窗口互相
   丢弃、无限重试（gqa+mla_prefill 在卡4 互相卡死 1 小时零进展）。**铁律：同卡
   任务必须同一 campaign 实例拉起**；单题补跑用 --tasks 单题在同实例内排队即可。
3. **kill 不彻底的孤儿**：kill bash 包装壳后 python 本体（ksearch-token-run.py）
   变孤儿继续跑，与新进程混写同一 run 目录（轮次双胞胎文件）+ 烧 token（34 分钟
   2.5 万输出 token，输入全命中缓存）。**杀任务必须 bash 壳 + python 本体按 PID
   逐一确认死亡**。
4. **同卡评测窗口的自踩竞态**：stranger_ok 先查 compute-apps 再查总显存，两次
   快照间自家评测 worker 新建上下文 >1GB 会被误判"不可见陌生占用"→ 窗口被自己
   踩死。缓解：KSEARCH_INVISIBLE_ALLOW_MB=6000；根治：跨卡分离任务（不同卡的任务
   在污染检测里天然互不可见）。
5. **nohup 随 shell 退出被杀**：交互 shell 里 `nohup ... &` 拉起的 campaign 在
   shell 退出后被杀（任务进程莫名消失的元凶之一）。长期进程必须
   `setsid nohup ... < /dev/null &`。
6. **归档上传 reset --soft 陷阱实测**：在 ksearch-h100/ 内部 git init 会把归档
   内容当仓库根，add -A 时远程其他实验线全被标 D（守卫拦截）。正确做法：上层
   目录 init（顶层只有单一实验子目录）→ fetch → reset --soft → 守卫查 D 标记
   → add 子目录。本次推送 a166c448，远程抽查 9 顶层项 + best_solutions 21 文件
   与本地一致。
