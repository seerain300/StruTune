# task: 049_group_limited_topk_routing
# bench: SOL-L2 | batch: formal2_solL2_20260916
# final eval (official evaluator, full workloads): valid=True pass=16/16 geomean=8.940x
# feedback best (5-workload sample during search): 10.673x
# torch fallback audit: 干净 (-)
# tokens: 1,773,814

import torch
import triton
import triton.language as tl


@triton.jit
def _dense_projection_compact_kernel(
    hidden_ptr,
    weight_ptr,
    scores_ptr,
    M,
    stride_hm: tl.constexpr,
    stride_hk: tl.constexpr,
    stride_we: tl.constexpr,
    stride_wk: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, 4096, BLOCK_K):
        k = k_start + offs_k
        hidden = tl.load(
            hidden_ptr
            + offs_m[:, None] * stride_hm
            + k[None, :] * stride_hk,
            mask=offs_m[:, None] < M,
            other=0.0,
        )
        weight = tl.load(
            weight_ptr
            + offs_n[:, None] * stride_we
            + k[None, :] * stride_wk,
        )
        accumulator += tl.dot(hidden, tl.trans(weight))

    scores = tl.sigmoid(accumulator)
    tl.store(
        scores_ptr + offs_m[:, None] * 256 + offs_n[None, :],
        scores,
        mask=offs_m[:, None] < M,
    )


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_K": 64},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_K": 32},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128},
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=["M"],
)
@triton.jit
def _dense_projection_kernel(
    hidden_ptr,
    weight_ptr,
    scores_ptr,
    M,
    stride_hm: tl.constexpr,
    stride_hk: tl.constexpr,
    stride_we: tl.constexpr,
    stride_wk: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, 4096, BLOCK_K):
        k = k_start + offs_k
        hidden = tl.load(
            hidden_ptr
            + offs_m[:, None] * stride_hm
            + k[None, :] * stride_hk,
            mask=offs_m[:, None] < M,
            other=0.0,
        )
        weight = tl.load(
            weight_ptr
            + offs_n[:, None] * stride_we
            + k[None, :] * stride_wk,
        )
        accumulator += tl.dot(hidden, tl.trans(weight))

    scores = tl.sigmoid(accumulator)
    tl.store(
        scores_ptr + offs_m[:, None] * 256 + offs_n[None, :],
        scores,
        mask=offs_m[:, None] < M,
    )


@triton.jit
def _routing_kernel(
    scores_ptr,
    bias_ptr,
    topk_idx_ptr,
    topk_weight_ptr,
    routed_scaling_factor,
):
    token = tl.program_id(0)
    expert = tl.arange(0, 256)
    scores_base = scores_ptr + token * 256

    scores = tl.load(scores_base + expert).to(tl.float32)
    bias = tl.load(bias_ptr + expert).to(tl.float32)
    routing_scores = scores + bias

    routing_by_group = tl.reshape(routing_scores, (8, 32))
    local_expert = tl.arange(0, 32)

    first_value, first_idx = tl.max(
        routing_by_group,
        axis=1,
        return_indices=True,
    )
    second_candidates = tl.where(
        local_expert[None, :] != first_idx[:, None],
        routing_by_group,
        -float("inf"),
    )
    second_value = tl.max(second_candidates, axis=1)
    group_scores = first_value + second_value

    group_ids = tl.arange(0, 8)
    other_is_better = (
        (group_scores[None, :] > group_scores[:, None])
        | (
            (group_scores[None, :] == group_scores[:, None])
            & (group_ids[None, :] < group_ids[:, None])
        )
    )
    group_rank = tl.sum(tl.where(other_is_better, 1, 0), axis=1)
    selected_groups = group_rank < 4

    eligible_by_group = (
        selected_groups[:, None]
        | tl.zeros((8, 32), dtype=tl.int1)
    )
    eligible_experts = tl.reshape(eligible_by_group, (256,))

    slots = tl.arange(0, 8)
    picked_indices = tl.zeros((8,), dtype=tl.int32)
    picked_routing_scores = tl.zeros((8,), dtype=tl.float32)

    expert_candidates = tl.where(
        eligible_experts,
        routing_scores,
        -float("inf"),
    )

    for slot in tl.static_range(0, 8):
        picked_routing_score, expert_idx = tl.max(
            expert_candidates,
            axis=0,
            return_indices=True,
        )
        expert_candidates = tl.where(
            expert == expert_idx,
            -float("inf"),
            expert_candidates,
        )
        picked_indices = tl.where(
            slots == slot,
            expert_idx,
            picked_indices,
        )
        picked_routing_scores = tl.where(
            slots == slot,
            picked_routing_score,
            picked_routing_scores,
        )

    threshold = picked_routing_score
    above_threshold = picked_routing_scores > threshold

    other_emits_first = (
        (
            above_threshold[None, :]
            & (~above_threshold[:, None])
        )
        | (
            (above_threshold[None, :] == above_threshold[:, None])
            & (picked_indices[None, :] < picked_indices[:, None])
        )
    )
    emission_rank = tl.sum(
        tl.where(other_emits_first, 1, 0),
        axis=1,
    )

    output_match = emission_rank[None, :] == slots[:, None]
    selected_indices = tl.sum(
        tl.where(output_match, picked_indices[None, :], 0),
        axis=1,
    )
    selected_values = tl.load(
        scores_base + selected_indices,
    ).to(tl.float32)

    denominator = tl.sum(selected_values, axis=0) + 1.0e-20
    normalized_weights = (
        selected_values / denominator
    ) * routed_scaling_factor

    output_offset = token * 8 + slots
    tl.store(topk_idx_ptr + output_offset, selected_indices)
    tl.store(topk_weight_ptr + output_offset, normalized_weights)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    expert_bias: torch.Tensor,
    routed_scaling_factor: float,
):
    num_tokens = hidden_states.shape[0]

    scores = torch.empty(
        (num_tokens, 256),
        device=hidden_states.device,
        dtype=torch.float16,
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

    if num_tokens == 0:
        return topk_idx, topk_weight

    with torch.cuda.device(hidden_states.device):
        if num_tokens <= 16:
            block_m = 16
            block_n = 32
            block_k = 128
            projection_grid = (
                triton.cdiv(num_tokens, block_m),
                triton.cdiv(256, block_n),
            )
            _dense_projection_compact_kernel[projection_grid](
                hidden_states,
                weight,
                scores,
                num_tokens,
                hidden_states.stride(0),
                hidden_states.stride(1),
                weight.stride(0),
                weight.stride(1),
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_warps=4,
                num_stages=4,
            )
        elif num_tokens <= 32:
            block_m = 16
            block_n = 64
            block_k = 64
            projection_grid = (
                triton.cdiv(num_tokens, block_m),
                triton.cdiv(256, block_n),
            )
            _dense_projection_compact_kernel[projection_grid](
                hidden_states,
                weight,
                scores,
                num_tokens,
                hidden_states.stride(0),
                hidden_states.stride(1),
                weight.stride(0),
                weight.stride(1),
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_warps=4,
                num_stages=4,
            )
        elif num_tokens <= 64:
            block_m = 32
            block_n = 64
            block_k = 64
            projection_grid = (
                triton.cdiv(num_tokens, block_m),
                triton.cdiv(256, block_n),
            )
            _dense_projection_compact_kernel[projection_grid](
                hidden_states,
                weight,
                scores,
                num_tokens,
                hidden_states.stride(0),
                hidden_states.stride(1),
                weight.stride(0),
                weight.stride(1),
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_warps=4,
                num_stages=4,
            )
        else:
            projection_grid = lambda meta: (
                triton.cdiv(num_tokens, meta["BLOCK_M"]),
                triton.cdiv(256, meta["BLOCK_N"]),
            )
            _dense_projection_kernel[projection_grid](
                hidden_states,
                weight,
                scores,
                num_tokens,
                hidden_states.stride(0),
                hidden_states.stride(1),
                weight.stride(0),
                weight.stride(1),
            )

        routing_num_warps = 8 if num_tokens <= 1024 else 4
        _routing_kernel[(num_tokens,)](
            scores,
            expert_bias,
            topk_idx,
            topk_weight,
            routed_scaling_factor,
            num_warps=routing_num_warps,
        )

    return topk_idx, topk_weight