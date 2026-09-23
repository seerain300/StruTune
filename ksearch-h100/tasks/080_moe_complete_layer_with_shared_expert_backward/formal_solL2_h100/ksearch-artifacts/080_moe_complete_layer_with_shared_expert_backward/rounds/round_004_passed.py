# solution=GPT-5.6-Sol_080_moe_complete_layer_with_shared_expert_backward_triton_optimized_r4 score=5.267542923823858 passed=True
import torch
import triton
import triton.language as tl


_HIDDEN_SIZE = 4096
_INTERMEDIATE_SIZE = 1408
_N_ROUTED_EXPERTS = 128


@triton.jit
def _shared_swiglu_backward_kernel(
    grad_activated_ptr,
    gate_output_ptr,
    up_output_ptr,
    grad_gate_up_ptr,
    intermediate_size: tl.constexpr,
    combined_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token = tl.program_id(0)
    feature_offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = feature_offsets < intermediate_size

    input_offset = token * intermediate_size + feature_offsets

    grad_activated = tl.load(
        grad_activated_ptr + input_offset,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        gate_output_ptr + input_offset,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        up_output_ptr + input_offset,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    sigmoid_gate = tl.sigmoid(gate)
    silu_gate = gate * sigmoid_gate

    grad_gate_silu = (grad_activated * up).to(tl.bfloat16)
    grad_up = (grad_activated * silu_gate).to(tl.bfloat16)

    grad_gate = (
        grad_gate_silu.to(tl.float32)
        * sigmoid_gate
        * (1.0 + gate * (1.0 - sigmoid_gate))
    ).to(tl.bfloat16)

    output_offset = token * combined_stride + feature_offsets

    tl.store(
        grad_gate_up_ptr + output_offset,
        grad_gate,
        mask=mask,
    )
    tl.store(
        grad_gate_up_ptr + output_offset + intermediate_size,
        grad_up,
        mask=mask,
    )


@triton.jit
def _shared_input_gradient_kernel(
    grad_gate_up_ptr,
    gate_weight_ptr,
    up_weight_ptr,
    grad_hidden_ptr,
    batch_seq_len,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(batch_seq_len, BLOCK_M)
    num_pid_n = tl.cdiv(hidden_size, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)

    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)

    grad_gate_ptrs = (
        grad_gate_up_ptr
        + offsets_m[:, None] * (2 * intermediate_size)
        + offsets_k[None, :]
    )
    grad_up_ptrs = grad_gate_ptrs + intermediate_size

    gate_weight_ptrs = (
        gate_weight_ptr
        + offsets_k[:, None] * hidden_size
        + offsets_n[None, :]
    )
    up_weight_ptrs = (
        up_weight_ptr
        + offsets_k[:, None] * hidden_size
        + offsets_n[None, :]
    )

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    row_mask = offsets_m[:, None] < batch_seq_len

    for _ in range(0, intermediate_size, BLOCK_K):
        grad_gate = tl.load(
            grad_gate_ptrs,
            mask=row_mask,
            other=0.0,
        )
        grad_up = tl.load(
            grad_up_ptrs,
            mask=row_mask,
            other=0.0,
        )
        gate_weight = tl.load(gate_weight_ptrs)
        up_weight = tl.load(up_weight_ptrs)

        accumulator += tl.dot(grad_gate, gate_weight)
        accumulator += tl.dot(grad_up, up_weight)

        grad_gate_ptrs += BLOCK_K
        grad_up_ptrs += BLOCK_K
        gate_weight_ptrs += BLOCK_K * hidden_size
        up_weight_ptrs += BLOCK_K * hidden_size

    output_ptrs = (
        grad_hidden_ptr
        + offsets_m[:, None] * hidden_size
        + offsets_n[None, :]
    )
    output_mask = (
        (offsets_m[:, None] < batch_seq_len)
        & (offsets_n[None, :] < hidden_size)
    )

    tl.store(
        output_ptrs,
        accumulator.to(tl.bfloat16),
        mask=output_mask,
    )


@triton.jit
def _router_gradient_prepare_kernel(
    grad_output_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    scores_ptr,
    grad_router_logits_ptr,
    grad_output_stride: tl.constexpr,
    topk_stride: tl.constexpr,
    router_stride: tl.constexpr,
    num_experts_per_tok: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    token = tl.program_id(0)

    hidden_offsets = tl.arange(0, BLOCK_H)
    grad_output = tl.load(
        grad_output_ptr + token * grad_output_stride + hidden_offsets
    ).to(tl.float32)
    grad_norm_sq = tl.sum(grad_output * grad_output, axis=0)
    grad_topk = grad_norm_sq / num_experts_per_tok

    topk_offsets = tl.arange(0, BLOCK_K)
    topk_mask = topk_offsets < num_experts_per_tok

    weights = tl.load(
        topk_weights_ptr + token * topk_stride + topk_offsets,
        mask=topk_mask,
        other=0.0,
    ).to(tl.float32)
    indices = tl.load(
        topk_indices_ptr + token * topk_stride + topk_offsets,
        mask=topk_mask,
        other=-1,
    )

    denominator = tl.sum(weights, axis=0) + 1.0e-20
    weighted_grad = tl.where(
        topk_mask,
        grad_topk * weights,
        0.0,
    )
    sum_grad = tl.sum(weighted_grad, axis=0) / denominator
    grad_before_norm = (grad_topk - sum_grad) / denominator

    expert_offsets = tl.arange(0, BLOCK_E)
    selected = expert_offsets[:, None] == indices[None, :]
    selected = selected & topk_mask[None, :]

    sparse_grad = tl.sum(
        tl.where(selected, grad_before_norm, 0.0),
        axis=1,
    )

    router_offsets = token * router_stride + expert_offsets
    score = tl.load(scores_ptr + router_offsets).to(tl.float32)

    grad_logits = sparse_grad * score * (1.0 - score)
    tl.store(
        grad_router_logits_ptr + router_offsets,
        grad_logits.to(tl.bfloat16),
    )


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    router_logits: torch.Tensor,
    scores: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    score_mask: torch.Tensor,
    shared_expert_gate_weight: torch.Tensor,
    shared_expert_up_weight: torch.Tensor,
    shared_expert_down_weight: torch.Tensor,
    shared_gate_output: torch.Tensor,
    shared_up_output: torch.Tensor,
    shared_activated: torch.Tensor,
):
    batch_seq_len = hidden_states.shape[0]
    num_experts_per_tok = topk_weights.shape[1]

    grad_shared_activated = torch.mm(
        grad_output,
        shared_expert_down_weight,
    )

    grad_shared_expert_down_weight = torch.mm(
        grad_output.transpose(0, 1),
        shared_activated,
    )

    grad_gate_up = torch.empty(
        (batch_seq_len, 2 * _INTERMEDIATE_SIZE),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )

    if batch_seq_len <= 384:
        swiglu_block_size = 2048
        swiglu_num_warps = 4
    elif batch_seq_len <= 512:
        swiglu_block_size = 2048
        swiglu_num_warps = 4
    elif batch_seq_len <= 1024:
        swiglu_block_size = 1024
        swiglu_num_warps = 4
    else:
        swiglu_block_size = 512
        swiglu_num_warps = 8

    _shared_swiglu_backward_kernel[
        (
            batch_seq_len,
            triton.cdiv(_INTERMEDIATE_SIZE, swiglu_block_size),
        )
    ](
        grad_shared_activated,
        shared_gate_output,
        shared_up_output,
        grad_gate_up,
        _INTERMEDIATE_SIZE,
        2 * _INTERMEDIATE_SIZE,
        BLOCK_SIZE=swiglu_block_size,
        num_warps=swiglu_num_warps,
    )

    grad_shared_weights = torch.mm(
        grad_gate_up.transpose(0, 1),
        hidden_states,
    )
    grad_shared_expert_gate_weight = grad_shared_weights[
        :_INTERMEDIATE_SIZE
    ]
    grad_shared_expert_up_weight = grad_shared_weights[
        _INTERMEDIATE_SIZE:
    ]

    grad_hidden_states = torch.empty_like(hidden_states)

    if batch_seq_len <= 512:
        block_m = 32
        block_n = 128
        block_k = 64
        group_m = 8
        input_num_warps = 8
        input_num_stages = 4
    elif batch_seq_len <= 2048:
        block_m = 64
        block_n = 64
        block_k = 32
        group_m = 8
        input_num_warps = 4
        input_num_stages = 4
    else:
        block_m = 128
        block_n = 128
        block_k = 64
        group_m = 8
        input_num_warps = 8
        input_num_stages = 3

    input_gradient_grid = (
        triton.cdiv(batch_seq_len, block_m)
        * triton.cdiv(_HIDDEN_SIZE, block_n),
    )

    _shared_input_gradient_kernel[input_gradient_grid](
        grad_gate_up,
        shared_expert_gate_weight,
        shared_expert_up_weight,
        grad_hidden_states,
        batch_seq_len,
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=_INTERMEDIATE_SIZE,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        num_warps=input_num_warps,
        num_stages=input_num_stages,
    )

    grad_router_logits = torch.empty(
        (batch_seq_len, _N_ROUTED_EXPERTS),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )

    _router_gradient_prepare_kernel[(batch_seq_len,)](
        grad_output,
        topk_indices,
        topk_weights,
        scores,
        grad_router_logits,
        grad_output.stride(0),
        topk_weights.stride(0),
        scores.stride(0),
        num_experts_per_tok,
        BLOCK_H=_HIDDEN_SIZE,
        BLOCK_K=8,
        BLOCK_E=_N_ROUTED_EXPERTS,
        num_warps=8,
    )

    grad_router_weight = torch.mm(
        grad_router_logits.transpose(0, 1),
        hidden_states,
        out_dtype=torch.float32,
    )

    return (
        grad_hidden_states,
        grad_router_weight,
        grad_shared_expert_gate_weight,
        grad_shared_expert_up_weight,
        grad_shared_expert_down_weight,
    )