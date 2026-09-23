# solution=GPT-5.6-Sol_049_group_limited_topk_routing_triton_optimized_r6 score=-1.0 passed=False
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128},
            num_warps=4,
            num_stages=4,
        ),
    ],
    key=["num_tokens"],
)
@triton.jit
def _routing_projection_kernel(
    hidden_states_ptr,
    weight_ptr,
    logits_ptr,
    num_tokens,
    gating_dim: tl.constexpr,
    hidden_stride_m,
    hidden_stride_k,
    weight_stride_e,
    weight_stride_k,
    logits_stride_m,
    logits_stride_e,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

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
            mask=(offs_m[:, None] < num_tokens)
            & (k_offsets[None, :] < gating_dim),
            other=0.0,
        )
        weights = tl.load(
            weight_ptrs,
            mask=(k_offsets[:, None] < gating_dim)
            & (offs_n[None, :] < 256),
            other=0.0,
        )

        acc = tl.dot(hidden, weights, acc=acc, out_dtype=tl.float32)

    logits_ptrs = (
        logits_ptr
        + offs_m[:, None] * logits_stride_m
        + offs_n[None, :] * logits_stride_e
    )
    tl.store(
        logits_ptrs,
        acc,
        mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < 256),
    )


@triton.jit
def _routing_postprocess_kernel(
    logits_ptr,
    expert_bias_ptr,
    topk_idx_ptr,
    topk_weight_ptr,
    routed_scaling_factor,
    num_tokens,
    logits_stride_m,
    logits_stride_e,
    idx_stride_m,
    idx_stride_k,
    out_stride_m,
    out_stride_k,
):
    token = tl.program_id(0)
    expert = tl.arange(0, 256)
    group = tl.arange(0, 8)
    lane = tl.arange(0, 32)

    logits = tl.load(
        logits_ptr + token * logits_stride_m + expert * logits_stride_e
    ).to(tl.float32)
    scores = tl.sigmoid(logits)
    bias = tl.load(expert_bias_ptr + expert).to(tl.float32)
    routing_scores = scores + bias

    grouped = tl.reshape(routing_scores, (8, 32))
    first_lane = tl.argmax(grouped, axis=1)
    first_value = tl.max(grouped, axis=1)
    second_value = tl.max(
        tl.where(lane[None, :] != first_lane[:, None], grouped, -float("inf")),
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
    selected_group = (
        (expert_group == group_0)
        | (expert_group == group_1)
        | (expert_group == group_2)
        | (expert_group == group_3)
    )
    candidates = tl.where(selected_group, routing_scores, -float("inf"))

    idx_0 = tl.argmax(candidates, axis=0)
    score_0 = tl.sum(tl.where(expert == idx_0, scores, 0.0), axis=0)
    candidates = tl.where(expert != idx_0, candidates, -float("inf"))

    idx_1 = tl.argmax(candidates, axis=0)
    score_1 = tl.sum(tl.where(expert == idx_1, scores, 0.0), axis=0)
    candidates = tl.where(expert != idx_1, candidates, -float("inf"))

    idx_2 = tl.argmax(candidates, axis=0)
    score_2 = tl.sum(tl.where(expert == idx_2, scores, 0.0), axis=0)
    candidates = tl.where(expert != idx_2, candidates, -float("inf"))

    idx_3 = tl.argmax(candidates, axis=0)
    score_3 = tl.sum(tl.where(expert == idx_3, scores, 0.0), axis=0)
    candidates = tl.where(expert != idx_3, candidates, -float("inf"))

    idx_4 = tl.argmax(candidates, axis=0)
    score_4 = tl.sum(tl.where(expert == idx_4, scores, 0.0), axis=0)
    candidates = tl.where(expert != idx_4, candidates, -float("inf"))

    idx_5 = tl.argmax(candidates, axis=0)
    score_5 = tl.sum(tl.where(expert == idx_5, scores, 0.0), axis=0)
    candidates = tl.where(expert != idx_5, candidates, -float("inf"))

    idx_6 = tl.argmax(candidates, axis=0)
    score_6 = tl.sum(tl.where(expert == idx_6, scores, 0.0), axis=0)
    candidates = tl.where(expert != idx_6, candidates, -float("inf"))

    idx_7 = tl.argmax(candidates, axis=0)
    score_7 = tl.sum(tl.where(expert == idx_7, scores, 0.0), axis=0)

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

    tl.store(idx_base + 0 * idx_stride_k, idx_0.to(tl.int64))
    tl.store(idx_base + 1 * idx_stride_k, idx_1.to(tl.int64))
    tl.store(idx_base + 2 * idx_stride_k, idx_2.to(tl.int64))
    tl.store(idx_base + 3 * idx_stride_k, idx_3.to(tl.int64))
    tl.store(idx_base + 4 * idx_stride_k, idx_4.to(tl.int64))
    tl.store(idx_base + 5 * idx_stride_k, idx_5.to(tl.int64))
    tl.store(idx_base + 6 * idx_stride_k, idx_6.to(tl.int64))
    tl.store(idx_base + 7 * idx_stride_k, idx_7.to(tl.int64))

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

    logits = torch.empty(
        (num_tokens, 256),
        device=hidden_states.device,
        dtype=torch.float32,
    )

    projection_grid = lambda meta: (
        triton.cdiv(num_tokens, meta["BLOCK_M"]),
        triton.cdiv(256, meta["BLOCK_N"]),
    )

    _routing_projection_kernel[projection_grid](
        hidden_states,
        weight,
        logits,
        num_tokens,
        gating_dim,
        hidden_states.stride(0),
        hidden_states.stride(1),
        weight.stride(0),
        weight.stride(1),
        logits.stride(0),
        logits.stride(1),
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
        logits,
        expert_bias,
        topk_idx,
        topk_weight,
        routed_scaling_factor,
        num_tokens,
        logits.stride(0),
        logits.stride(1),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_weight.stride(0),
        topk_weight.stride(1),
        num_warps=8,
    )

    return topk_idx, topk_weight