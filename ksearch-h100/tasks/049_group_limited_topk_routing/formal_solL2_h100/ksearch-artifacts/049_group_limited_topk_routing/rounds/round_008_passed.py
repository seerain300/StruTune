# solution=GPT-5.6-Sol_049_group_limited_topk_routing_triton_optimized_r8 score=6.704443860545978 passed=True
import torch
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
            weights = tl.load(weight_ptrs, eviction_policy="evict_last")
            acc = tl.dot(hidden, weights, acc=acc, out_dtype=tl.float32)

        scores_ptrs = (
            scores_ptr
            + offs_m[:, None] * scores_stride_m
            + offs_n[None, :] * scores_stride_e
        )
        tl.store(
            scores_ptrs,
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
    remaining_groups = tl.where(
        group != group_0,
        group_scores,
        -float("inf"),
    )
    group_1 = tl.argmax(remaining_groups, axis=0)

    remaining_groups = tl.where(
        group != group_1,
        remaining_groups,
        -float("inf"),
    )
    group_2 = tl.argmax(remaining_groups, axis=0)

    remaining_groups = tl.where(
        group != group_2,
        remaining_groups,
        -float("inf"),
    )
    group_3 = tl.argmax(remaining_groups, axis=0)

    candidate_lane = tl.arange(0, 128)
    candidate_group = tl.where(
        candidate_lane < 32,
        group_0,
        tl.where(
            candidate_lane < 64,
            group_1,
            tl.where(candidate_lane < 96, group_2, group_3),
        ),
    )
    candidate_expert = candidate_group * 32 + (candidate_lane & 31)

    candidate_scores = tl.load(
        scores_base + candidate_expert * scores_stride_e
    ).to(tl.float32)
    candidate_bias = tl.load(
        expert_bias_ptr + candidate_expert
    ).to(tl.float32)
    candidates = candidate_scores + candidate_bias

    rank_lane_0 = tl.argmax(candidates, axis=0)
    rank_0 = tl.where(
        rank_lane_0 < 32,
        group_0,
        tl.where(
            rank_lane_0 < 64,
            group_1,
            tl.where(rank_lane_0 < 96, group_2, group_3),
        ),
    ) * 32 + (rank_lane_0 & 31)
    candidates = tl.where(
        candidate_lane != rank_lane_0,
        candidates,
        -float("inf"),
    )

    rank_lane_1 = tl.argmax(candidates, axis=0)
    rank_1 = tl.where(
        rank_lane_1 < 32,
        group_0,
        tl.where(
            rank_lane_1 < 64,
            group_1,
            tl.where(rank_lane_1 < 96, group_2, group_3),
        ),
    ) * 32 + (rank_lane_1 & 31)
    candidates = tl.where(
        candidate_lane != rank_lane_1,
        candidates,
        -float("inf"),
    )

    rank_lane_2 = tl.argmax(candidates, axis=0)
    rank_2 = tl.where(
        rank_lane_2 < 32,
        group_0,
        tl.where(
            rank_lane_2 < 64,
            group_1,
            tl.where(rank_lane_2 < 96, group_2, group_3),
        ),
    ) * 32 + (rank_lane_2 & 31)
    candidates = tl.where(
        candidate_lane != rank_lane_2,
        candidates,
        -float("inf"),
    )

    rank_lane_3 = tl.argmax(candidates, axis=0)
    rank_3 = tl.where(
        rank_lane_3 < 32,
        group_0,
        tl.where(
            rank_lane_3 < 64,
            group_1,
            tl.where(rank_lane_3 < 96, group_2, group_3),
        ),
    ) * 32 + (rank_lane_3 & 31)
    candidates = tl.where(
        candidate_lane != rank_lane_3,
        candidates,
        -float("inf"),
    )

    rank_lane_4 = tl.argmax(candidates, axis=0)
    rank_4 = tl.where(
        rank_lane_4 < 32,
        group_0,
        tl.where(
            rank_lane_4 < 64,
            group_1,
            tl.where(rank_lane_4 < 96, group_2, group_3),
        ),
    ) * 32 + (rank_lane_4 & 31)
    candidates = tl.where(
        candidate_lane != rank_lane_4,
        candidates,
        -float("inf"),
    )

    rank_lane_5 = tl.argmax(candidates, axis=0)
    rank_5 = tl.where(
        rank_lane_5 < 32,
        group_0,
        tl.where(
            rank_lane_5 < 64,
            group_1,
            tl.where(rank_lane_5 < 96, group_2, group_3),
        ),
    ) * 32 + (rank_lane_5 & 31)
    candidates = tl.where(
        candidate_lane != rank_lane_5,
        candidates,
        -float("inf"),
    )

    rank_lane_6 = tl.argmax(candidates, axis=0)
    rank_6 = tl.where(
        rank_lane_6 < 32,
        group_0,
        tl.where(
            rank_lane_6 < 64,
            group_1,
            tl.where(rank_lane_6 < 96, group_2, group_3),
        ),
    ) * 32 + (rank_lane_6 & 31)
    candidates = tl.where(
        candidate_lane != rank_lane_6,
        candidates,
        -float("inf"),
    )

    rank_lane_7 = tl.argmax(candidates, axis=0)
    rank_7 = tl.where(
        rank_lane_7 < 32,
        group_0,
        tl.where(
            rank_lane_7 < 64,
            group_1,
            tl.where(rank_lane_7 < 96, group_2, group_3),
        ),
    ) * 32 + (rank_lane_7 & 31)

    selected_0 = rank_0
    selected_1 = rank_1
    selected_2 = rank_2
    selected_3 = rank_3
    selected_4 = rank_4
    selected_5 = rank_5
    selected_6 = rank_6
    selected_7 = 256

    lo = tl.minimum(selected_0, selected_1)
    hi = tl.maximum(selected_0, selected_1)
    selected_0, selected_1 = lo, hi

    lo = tl.minimum(selected_2, selected_3)
    hi = tl.maximum(selected_2, selected_3)
    selected_2, selected_3 = lo, hi

    lo = tl.minimum(selected_4, selected_5)
    hi = tl.maximum(selected_4, selected_5)
    selected_4, selected_5 = lo, hi

    lo = tl.minimum(selected_6, selected_7)
    hi = tl.maximum(selected_6, selected_7)
    selected_6, selected_7 = lo, hi

    lo = tl.minimum(selected_0, selected_2)
    hi = tl.maximum(selected_0, selected_2)
    selected_0, selected_2 = lo, hi

    lo = tl.minimum(selected_1, selected_3)
    hi = tl.maximum(selected_1, selected_3)
    selected_1, selected_3 = lo, hi

    lo = tl.minimum(selected_4, selected_6)
    hi = tl.maximum(selected_4, selected_6)
    selected_4, selected_6 = lo, hi

    lo = tl.minimum(selected_5, selected_7)
    hi = tl.maximum(selected_5, selected_7)
    selected_5, selected_7 = lo, hi

    lo = tl.minimum(selected_1, selected_2)
    hi = tl.maximum(selected_1, selected_2)
    selected_1, selected_2 = lo, hi

    lo = tl.minimum(selected_5, selected_6)
    hi = tl.maximum(selected_5, selected_6)
    selected_5, selected_6 = lo, hi

    lo = tl.minimum(selected_0, selected_4)
    hi = tl.maximum(selected_0, selected_4)
    selected_0, selected_4 = lo, hi

    lo = tl.minimum(selected_3, selected_7)
    hi = tl.maximum(selected_3, selected_7)
    selected_3, selected_7 = lo, hi

    lo = tl.minimum(selected_1, selected_5)
    hi = tl.maximum(selected_1, selected_5)
    selected_1, selected_5 = lo, hi

    lo = tl.minimum(selected_2, selected_6)
    hi = tl.maximum(selected_2, selected_6)
    selected_2, selected_6 = lo, hi

    lo = tl.minimum(selected_1, selected_4)
    hi = tl.maximum(selected_1, selected_4)
    selected_1, selected_4 = lo, hi

    lo = tl.minimum(selected_3, selected_6)
    hi = tl.maximum(selected_3, selected_6)
    selected_3, selected_6 = lo, hi

    lo = tl.minimum(selected_2, selected_4)
    hi = tl.maximum(selected_2, selected_4)
    selected_2, selected_4 = lo, hi

    lo = tl.minimum(selected_3, selected_5)
    hi = tl.maximum(selected_3, selected_5)
    selected_3, selected_5 = lo, hi

    lo = tl.minimum(selected_3, selected_4)
    hi = tl.maximum(selected_3, selected_4)
    selected_3, selected_4 = lo, hi

    selected_7 = rank_7

    score_0 = tl.load(
        scores_base + selected_0 * scores_stride_e
    ).to(tl.float32)
    score_1 = tl.load(
        scores_base + selected_1 * scores_stride_e
    ).to(tl.float32)
    score_2 = tl.load(
        scores_base + selected_2 * scores_stride_e
    ).to(tl.float32)
    score_3 = tl.load(
        scores_base + selected_3 * scores_stride_e
    ).to(tl.float32)
    score_4 = tl.load(
        scores_base + selected_4 * scores_stride_e
    ).to(tl.float32)
    score_5 = tl.load(
        scores_base + selected_5 * scores_stride_e
    ).to(tl.float32)
    score_6 = tl.load(
        scores_base + selected_6 * scores_stride_e
    ).to(tl.float32)
    score_7 = tl.load(
        scores_base + selected_7 * scores_stride_e
    ).to(tl.float32)

    score_sum = (
        score_0
        + score_1
        + score_2
        + score_3
        + score_4
        + score_5
        + score_6
        + score_7
        + 1.0e-20
    )
    scale = routed_scaling_factor / score_sum

    idx_base = topk_idx_ptr + token * idx_stride_m
    weight_base = topk_weight_ptr + token * out_stride_m

    tl.store(idx_base + 0 * idx_stride_k, selected_0.to(tl.int64))
    tl.store(idx_base + 1 * idx_stride_k, selected_1.to(tl.int64))
    tl.store(idx_base + 2 * idx_stride_k, selected_2.to(tl.int64))
    tl.store(idx_base + 3 * idx_stride_k, selected_3.to(tl.int64))
    tl.store(idx_base + 4 * idx_stride_k, selected_4.to(tl.int64))
    tl.store(idx_base + 5 * idx_stride_k, selected_5.to(tl.int64))
    tl.store(idx_base + 6 * idx_stride_k, selected_6.to(tl.int64))
    tl.store(idx_base + 7 * idx_stride_k, selected_7.to(tl.int64))

    tl.store(weight_base + 0 * out_stride_k, score_0 * scale)
    tl.store(weight_base + 1 * out_stride_k, score_1 * scale)
    tl.store(weight_base + 2 * out_stride_k, score_2 * scale)
    tl.store(weight_base + 3 * out_stride_k, score_3 * scale)
    tl.store(weight_base + 4 * out_stride_k, score_4 * scale)
    tl.store(weight_base + 5 * out_stride_k, score_5 * scale)
    tl.store(weight_base + 6 * out_stride_k, score_6 * scale)
    tl.store(weight_base + 7 * out_stride_k, score_7 * scale)


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

    projection_ctas = 396 if num_tokens >= 8192 else 264

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