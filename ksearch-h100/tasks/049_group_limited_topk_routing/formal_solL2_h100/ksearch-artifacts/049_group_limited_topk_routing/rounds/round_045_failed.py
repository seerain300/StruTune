# solution=GPT-5.6-Sol_049_group_limited_topk_routing_triton_optimized_r45 score=-1.0 passed=False
I’m applying a narrow postprocess optimization: keep the resident score vector and use a direct reduction for each selected expert’s unbiased score, avoiding the per-rank masked sum while leaving projection, selection order, and output layout unchanged.import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=2),
    ],
    key=["num_tokens"],
)
@triton.jit
def _routing_projection_kernel(
    hidden_states_ptr,
    weight_ptr,
    scores_ptr,
    num_tokens,
    gating_dim: tl.constexpr,
    hidden_stride_m: tl.constexpr,
    hidden_stride_k: tl.constexpr,
    weight_stride_e: tl.constexpr,
    weight_stride_k: tl.constexpr,
    scores_stride_m: tl.constexpr,
    scores_stride_e: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    num_pid_m = tl.cdiv(num_tokens, BLOCK_M)
    num_pid_n = tl.cdiv(256, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    group_m: tl.constexpr = 8

    tile_id = pid
    while tile_id < num_tiles:
        group_id = tile_id // (group_m * num_pid_n)
        group_start_m = group_id * group_m
        group_size_m = tl.minimum(group_m, num_pid_m - group_start_m)

        tile_in_group = tile_id - group_id * group_m * num_pid_n
        pid_m = group_start_m + tile_in_group % group_size_m
        pid_n = tile_in_group // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, gating_dim, BLOCK_K):
            k_offsets = k_start + offs_k

            hidden_ptrs = (
                hidden_states_ptr
                + offs_m[:, None] * hidden_stride_m
                + k_offsets[None, :] * hidden_stride_k
            )
            weight_ptrs = (
                weight_ptr
                + k_offsets[:, None] * weight_stride_k
                + offs_n[None, :] * weight_stride_e
            )

            hidden = tl.load(
                hidden_ptrs,
                mask=offs_m[:, None] < num_tokens,
                other=0.0,
                eviction_policy="evict_first",
            )
            weights = tl.load(
                weight_ptrs,
                eviction_policy="evict_last",
            )
            acc = tl.dot(hidden, weights, acc=acc, out_dtype=tl.float32)

        score_ptrs = (
            scores_ptr
            + offs_m[:, None] * scores_stride_m
            + offs_n[None, :] * scores_stride_e
        )
        tl.store(
            score_ptrs,
            tl.sigmoid(acc),
            mask=offs_m[:, None] < num_tokens,
        )

        tile_id += num_programs


@triton.jit
def _routing_postprocess_kernel(
    scores_ptr,
    expert_bias_ptr,
    topk_idx_ptr,
    topk_weight_ptr,
    routed_scaling_factor,
    scores_stride_m: tl.constexpr,
    scores_stride_e: tl.constexpr,
    idx_stride_m: tl.constexpr,
    idx_stride_k: tl.constexpr,
    out_stride_m: tl.constexpr,
    out_stride_k: tl.constexpr,
):
    token = tl.program_id(0)

    expert = tl.arange(0, 256)
    group = tl.arange(0, 8)
    lane = tl.arange(0, 32)
    rank_slot = tl.arange(0, 8)

    scores_base = scores_ptr + token * scores_stride_m
    scores = tl.load(scores_base + expert * scores_stride_e).to(tl.float32)
    bias = tl.load(expert_bias_ptr + expert).to(tl.float32)
    routing_scores = scores + bias

    grouped = tl.reshape(routing_scores, (8, 32))
    first_lane = tl.argmax(grouped, axis=1)
    first_value = tl.max(grouped, axis=1)
    second_value = tl.max(
        tl.where(
            lane[None, :] != first_lane[:, None],
            grouped,
            -float("inf"),
        ),
        axis=1,
    )
    group_scores = first_value + second_value

    group_0 = tl.argmax(group_scores, axis=0)
    remaining_groups = tl.where(group != group_0, group_scores, -float("inf"))

    group_1 = tl.argmax(remaining_groups, axis=0)
    remaining_groups = tl.where(group != group_1, remaining_groups, -float("inf"))

    group_2 = tl.argmax(remaining_groups, axis=0)
    remaining_groups = tl.where(group != group_2, remaining_groups, -float("inf"))

    group_3 = tl.argmax(remaining_groups, axis=0)

    expert_group = expert // 32
    selected_group_mask = (
        (expert_group == group_0)
        | (expert_group == group_1)
        | (expert_group == group_2)
        | (expert_group == group_3)
    )

    candidates = tl.where(
        selected_group_mask,
        routing_scores,
        -float("inf"),
    )

    rank_experts = tl.zeros((8,), dtype=tl.int32)
    rank_scores = tl.zeros((8,), dtype=tl.float32)

    for rank in tl.static_range(0, 8):
        selected_expert = tl.argmax(candidates, axis=0)
        selected_score = tl.max(
            tl.where(
                expert == selected_expert,
                scores,
                -float("inf"),
            ),
            axis=0,
        )

        rank_experts = tl.where(
            rank_slot == rank,
            selected_expert,
            rank_experts,
        )
        rank_scores = tl.where(
            rank_slot == rank,
            selected_score,
            rank_scores,
        )
        candidates = tl.where(
            expert != selected_expert,
            candidates,
            -float("inf"),
        )

    rank_7 = tl.sum(
        tl.where(rank_slot == 7, rank_experts, 0),
        axis=0,
    )
    selected = tl.sort(
        tl.where(rank_slot == 7, 256, rank_experts),
        dim=0,
    )
    selected = tl.where(rank_slot == 7, rank_7, selected)

    selected_scores = tl.sum(
        tl.where(
            selected[:, None] == rank_experts[None, :],
            rank_scores[None, :],
            0.0,
        ),
        axis=1,
    )
    score_sum = tl.sum(rank_scores, axis=0) + 1.0e-20
    selected_weights = selected_scores * (
        routed_scaling_factor / score_sum
    )

    idx_base = topk_idx_ptr + token * idx_stride_m
    weight_base = topk_weight_ptr + token * out_stride_m

    tl.store(
        idx_base + rank_slot * idx_stride_k,
        selected.to(tl.int64),
    )
    tl.store(
        weight_base + rank_slot * out_stride_k,
        selected_weights,
    )


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    expert_bias: torch.Tensor,
    routed_scaling_factor: float,
):
    num_tokens = hidden_states.shape[0]
    gating_dim = hidden_states.shape[1]

    scores = torch.empty(
        (num_tokens, 256),
        device=hidden_states.device,
        dtype=torch.float16,
    )

    if num_tokens <= 4096:
        projection_ctas = 132
    elif num_tokens <= 12288:
        projection_ctas = 264
    else:
        projection_ctas = 396

    projection_grid = lambda meta: (
        min(
            triton.cdiv(num_tokens, meta["BLOCK_M"])
            * triton.cdiv(256, meta["BLOCK_N"]),
            projection_ctas,
        ),
    )

    _routing_projection_kernel[projection_grid](
        hidden_states,
        weight,
        scores,
        num_tokens,
        gating_dim,
        hidden_states.stride(0),
        hidden_states.stride(1),
        weight.stride(0),
        weight.stride(1),
        scores.stride(0),
        scores.stride(1),
    )

    topk_idx = torch.empty(
        (num_tokens, 8),
        device=hidden_states.device,
        dtype=torch.int64,
    )
    topk_weight = torch.empty(
        (num_tokens, 8),
        device=hidden_states.device,
        dtype=torch.float32,
    )

    _routing_postprocess_kernel[(num_tokens,)](
        scores,
        expert_bias,
        topk_idx,
        topk_weight,
        routed_scaling_factor,
        scores.stride(0),
        scores.stride(1),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_weight.stride(0),
        topk_weight.stride(1),
        num_warps=4,
    )

    return topk_idx, topk_weight