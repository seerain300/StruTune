# solution=GPT-5.6-Sol_080_moe_complete_layer_with_shared_expert_backward_triton_optimized_r51 score=4.26931132486259 passed=True
import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_backward_kernel(
    grad_activated_ptr,
    gate_ptr,
    up_ptr,
    grad_gate_up_ptr,
    n_rows,
    INTERMEDIATE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_ROW: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // BLOCKS_PER_ROW
    block_col = pid - row * BLOCKS_PER_ROW
    cols = block_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (row < n_rows) & (cols < INTERMEDIATE_SIZE)

    offsets = row * INTERMEDIATE_SIZE + cols
    grad_activated = tl.load(
        grad_activated_ptr + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        gate_ptr + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        up_ptr + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    sigmoid_gate = tl.sigmoid(gate)
    silu_gate = (gate * sigmoid_gate).to(tl.bfloat16)
    grad_gate_silu = (grad_activated * up).to(tl.bfloat16)

    grad_up = grad_activated * silu_gate.to(tl.float32)
    silu_derivative = sigmoid_gate * (
        1.0 + gate * (1.0 - sigmoid_gate)
    )
    grad_gate = grad_gate_silu.to(tl.float32) * silu_derivative

    output_row = row * (2 * INTERMEDIATE_SIZE)
    tl.store(
        grad_gate_up_ptr + output_row + cols,
        grad_gate,
        mask=mask,
    )
    tl.store(
        grad_gate_up_ptr + output_row + INTERMEDIATE_SIZE + cols,
        grad_up,
        mask=mask,
    )


@triton.jit
def _router_backward_prep_kernel(
    grad_output_ptr,
    scores_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    score_mask_ptr,
    grad_router_logits_ptr,
    grad_router_logits_bf16_ptr,
    HIDDEN_SIZE: tl.constexpr,
    N_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    EXPERT_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)

    hidden_cols = tl.arange(0, HIDDEN_BLOCK)
    hidden_mask = hidden_cols < HIDDEN_SIZE
    grad_output = tl.load(
        grad_output_ptr + row * HIDDEN_SIZE + hidden_cols,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)
    grad_norm_sq = tl.sum(grad_output * grad_output, axis=0)
    grad_topk_weight = grad_norm_sq / TOP_K

    topk_cols = tl.arange(0, TOP_K)
    topk_weights = tl.load(
        topk_weights_ptr + row * TOP_K + topk_cols,
    ).to(tl.float32)
    denominator = tl.sum(topk_weights, axis=0) + 1.0e-20
    sum_grad = tl.sum(
        grad_topk_weight * topk_weights,
        axis=0,
    ) / denominator
    grad_before_norm = (
        grad_topk_weight - sum_grad
    ) / denominator

    expert_cols = tl.arange(0, EXPERT_BLOCK)
    expert_mask = expert_cols < N_EXPERTS
    selected = tl.zeros((EXPERT_BLOCK,), dtype=tl.int1)

    for k in range(0, TOP_K):
        expert_index = tl.load(
            topk_indices_ptr + row * TOP_K + k,
        ).to(tl.int32)
        selected = selected | (expert_cols == expert_index)

    expert_offsets = row * N_EXPERTS + expert_cols
    selected_mask = expert_mask & selected
    score = tl.load(
        scores_ptr + expert_offsets,
        mask=selected_mask,
        other=0.0,
    ).to(tl.float32)
    group_mask = tl.load(
        score_mask_ptr + expert_offsets,
        mask=selected_mask,
        other=0.0,
    ).to(tl.float32)

    grad_router_logits = tl.where(
        selected,
        grad_before_norm * group_mask * score * (1.0 - score),
        0.0,
    )
    tl.store(
        grad_router_logits_ptr + expert_offsets,
        grad_router_logits,
        mask=expert_mask,
    )
    tl.store(
        grad_router_logits_bf16_ptr + expert_offsets,
        grad_router_logits,
        mask=expert_mask,
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
    hidden_size = hidden_states.shape[1]
    intermediate_size = shared_gate_output.shape[1]
    n_experts = router_weight.shape[0]
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
        (batch_seq_len, 2 * intermediate_size),
        dtype=shared_gate_output.dtype,
        device=shared_gate_output.device,
    )

    block_size = 1024
    blocks_per_row = triton.cdiv(intermediate_size, block_size)
    _swiglu_backward_kernel[
        (batch_seq_len * blocks_per_row,)
    ](
        grad_shared_activated,
        shared_gate_output,
        shared_up_output,
        grad_gate_up,
        batch_seq_len,
        INTERMEDIATE_SIZE=intermediate_size,
        BLOCK_SIZE=block_size,
        BLOCKS_PER_ROW=blocks_per_row,
        num_warps=8,
    )

    grad_shared_gate_output = grad_gate_up[:, :intermediate_size]
    grad_shared_up_output = grad_gate_up[:, intermediate_size:]

    grad_gate_up_weight = torch.mm(
        grad_gate_up.transpose(0, 1),
        hidden_states,
    )
    grad_shared_expert_gate_weight = grad_gate_up_weight[:intermediate_size]
    grad_shared_expert_up_weight = grad_gate_up_weight[intermediate_size:]

    grad_hidden_states = torch.mm(
        grad_shared_up_output,
        shared_expert_up_weight,
    )
    torch.addmm(
        grad_hidden_states,
        grad_shared_gate_output,
        shared_expert_gate_weight,
        out=grad_hidden_states,
    )

    grad_router_logits = torch.empty(
        (batch_seq_len, n_experts),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    grad_router_logits_bf16 = torch.empty(
        (batch_seq_len, n_experts),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    _router_backward_prep_kernel[(batch_seq_len,)](
        grad_output,
        scores,
        topk_indices,
        topk_weights,
        score_mask,
        grad_router_logits,
        grad_router_logits_bf16,
        HIDDEN_SIZE=hidden_size,
        N_EXPERTS=n_experts,
        TOP_K=num_experts_per_tok,
        HIDDEN_BLOCK=triton.next_power_of_2(hidden_size),
        EXPERT_BLOCK=triton.next_power_of_2(n_experts),
        num_warps=8,
    )

    torch.addmm(
        grad_hidden_states,
        grad_router_logits_bf16,
        router_weight,
        out=grad_hidden_states,
    )
    grad_router_weight = torch.mm(
        grad_router_logits.transpose(0, 1),
        hidden_states.to(torch.float32),
    )

    return (
        grad_hidden_states,
        grad_router_weight,
        grad_shared_expert_gate_weight,
        grad_shared_expert_up_weight,
        grad_shared_expert_down_weight,
    )