# solution=GPT-5.6-Sol_049_group_limited_topk_routing_triton_optimized_r7 score=-1.0 passed=False
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

    remaining_groups = group_scores
    group_0 = tl.argmax(remaining_groups, axis=0)
    remaining_groups = tl.where(group != group_0, remaining_groups, -float("inf"))

    group_1 = tl.argmax(remaining_groups, axis=0)
    remaining_groups = tl.where(group != group_1, remaining_groups, -float("inf"))

    group_2 = tl.argmax(remaining_groups, axis=0)
    remaining_groups = tl.where(group != group_2, remaining_groups, -float("inf"))

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

    output_lane = tl.arange(0, 8)
    selected = tl.zeros((8,), dtype=tl.int32)

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
    selected = tl.where(output_lane == 0, rank_0, selected)
    candidates = tl.where(candidate_lane != rank_lane_0, candidates, -float("inf"))

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
    selected = tl.where(output_lane == 1, rank_1, selected)
    candidates = tl.where(candidate_lane != rank_lane_1, candidates, -float("inf"))

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
    selected = tl.where(output_lane == 2, rank_2, selected)
    candidates = tl.where(candidate_lane != rank_lane_2, candidates, -float("inf"))

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
    selected = tl.where(output_lane == 3, rank_3, selected)
    candidates = tl.where(candidate_lane != rank_lane_3, candidates, -float("inf"))

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
    selected = tl.where(output_lane == 4, rank_4, selected)
    candidates = tl.where(candidate_lane != rank_lane_4, candidates, -float("inf"))

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
    selected = tl.where(output_lane == 5, rank_5, selected)
    candidates = tl.where(candidate_lane != rank_lane_5, candidates, -float("inf"))

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
    selected = tl.where(output_lane == 6, rank_6, selected)
    candidates = tl.where(candidate_lane != rank_lane_6, candidates, -float("inf"))

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
    selected = tl.where(output_lane == 7, rank_7, selected)

    lo = tl.minimum(selected[0], selected[1])
    hi = tl.maximum(selected[0], selected[1])
    selected = tl.where(output_lane == 0, lo, tl.where(output_lane == 1, hi, selected))

    lo = tl.minimum(selected[2], selected[3])
    hi = tl.maximum(selected[2], selected[3])
    selected = tl.where(output_lane == 2, lo, tl.where(output_lane == 3, hi, selected))

    lo = tl.minimum(selected[4], selected[5])
    hi = tl.maximum(selected[4], selected[5])
    selected = tl.where(output_lane == 4, lo, tl.where(output_lane == 5, hi, selected))

    lo = tl.minimum(selected[6], selected[7])
    hi = tl.maximum(selected[6], selected[7])
    selected = tl.where(output_lane == 6, lo, tl.where(output_lane == 7, hi, selected))

    lo = tl.minimum(selected[0], selected[2])
    hi = tl.maximum(selected[0], selected[2])
    selected = tl.where(output_lane == 0, lo, tl.where(output_lane == 2, hi, selected))

    lo = tl.minimum(selected[1], selected[3])
    hi = tl.maximum(selected[1], selected[3])
    selected = tl.where(output_lane == 1, lo, tl.where(output_lane == 3, hi, selected))

    lo = tl.minimum(selected[4], selected[6])
    hi = tl.maximum(selected[4], selected[6])
    selected = tl.where(output_lane == 4, lo, tl.where(output_lane == 6, hi, selected))

    lo = tl.minimum(selected[5], selected[7])
    hi = tl.maximum(selected[5], selected[7])
    selected = tl.where(output_lane == 5, lo, tl.where(output_lane == 7, hi, selected))

    lo = tl.minimum(selected[1], selected[2])
    hi = tl.maximum(selected[1], selected[2])
    selected = tl.where(output_lane == 1, lo, tl.where(output_lane == 2, hi, selected))

    lo = tl.minimum(selected[5], selected[6])
    hi = tl.maximum(selected[5], selected[6])
    selected = tl.where(output_lane == 5, lo, tl.where(output_lane == 6, hi, selected))

    lo = tl.minimum(selected[0], selected[4])
    hi = tl.maximum(selected[0], selected[4])
    selected = tl.where(output_lane == 0, lo, tl.where(output_lane == 4, hi, selected))

    lo = tl.minimum(selected[3], selected[7])
    hi = tl.maximum(selected[3], selected[7])
    selected = tl.where(output_lane == 3, lo, tl.where(output_lane == 7, hi, selected))

    lo = tl.minimum(selected[1], selected[5])
    hi = tl.maximum(selected[1], selected[5])
    selected = tl.where(output_lane == 1, lo, tl.where(output_lane == 5, hi, selected))

    lo = tl.minimum(selected[2], selected[6])
    hi = tl.maximum(selected[2], selected[6])
    selected = tl.where(output_lane == 2, lo, tl.where(output_lane == 6, hi, selected))

    lo = tl.minimum(selected[1], selected[4])
    hi = tl.maximum(selected[1], selected[4])
    selected = tl.where(output_lane == 1, lo, tl.where(output_lane == 4, hi, selected))

    lo = tl.minimum(selected[3], selected[6])
    hi = tl.maximum(selected[3], selected[6])
    selected = tl.where(output_lane == 3, lo, tl.where(output_lane == 6, hi, selected))

    lo = tl.minimum(selected[2], selected[4])
    hi = tl.maximum(selected[2], selected[4])
    selected = tl.where(output_lane == 2, lo, tl.where(output_lane == 4, hi, selected))

    lo = tl.minimum(selected[3], selected[5])
    hi = tl.maximum(selected[3], selected[5])
    selected = tl.where(output_lane == 3, lo, tl.where(output_lane == 5, hi, selected))

    lo = tl.minimum(selected[3], selected[4])
    hi = tl.maximum(selected[3], selected[4])
    selected = tl.where(output_lane == 3, lo, tl.where(output_lane == 4, hi, selected))

    selected = tl.where(output_lane == 7, rank_7, selected)

    selected_scores = tl.load(
        scores_base[:, None] + selected[None, :] * scores_stride_e
    ).to(tl.float32)
    score_sum = tl.sum(selected_scores, axis=0) + 1.0e-20
    scale = routed_scaling_factor / score_sum

    idx_base = topk_idx_ptr + token * idx_stride_m
    weight_base = topk_weight_ptr + token * out_stride_m

    tl.store(
        idx_base + output_lane * idx_stride_k,
        selected.to(tl.int64),
    )
    tl.store(
        weight_base + output_lane * out_stride_k,
        selected_scores * scale,
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