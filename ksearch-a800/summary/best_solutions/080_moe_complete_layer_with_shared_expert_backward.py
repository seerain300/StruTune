# task: 080_moe_complete_layer_with_shared_expert_backward
# bench: SOL-L2 | batch: formal2_solL2_20260916
# final eval (official evaluator, full workloads): valid=True pass=16/16 geomean=6.027x
# feedback best (5-workload sample during search): 4.831x
# torch fallback audit: B·自研为主 (matmul×7,addmm×2)
# tokens: 1,368,648

import torch
import triton
import triton.language as tl


@triton.jit
def _shared_swiglu_backward_paired_kernel(
    grad_activated_ptr,
    gate_ptr,
    up_ptr,
    grad_pair_ptr,
    INTERMEDIATE_SIZE: tl.constexpr,
    PAIR_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < INTERMEDIATE_SIZE

    input_offsets = token * INTERMEDIATE_SIZE + offsets
    output_offsets = token * PAIR_SIZE + offsets

    grad_activated = tl.load(
        grad_activated_ptr + input_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        gate_ptr + input_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        up_ptr + input_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    grad_gate_silu = (
        grad_activated * up
    ).to(tl.bfloat16).to(tl.float32)

    sigmoid_gate = tl.sigmoid(gate)
    silu_gate = (
        gate * sigmoid_gate
    ).to(tl.bfloat16).to(tl.float32)

    grad_up = (
        grad_activated * silu_gate
    ).to(tl.bfloat16)

    silu_derivative = sigmoid_gate * (
        1.0 + gate * (1.0 - sigmoid_gate)
    )
    grad_gate = (
        grad_gate_silu * silu_derivative
    ).to(tl.bfloat16)

    tl.store(
        grad_pair_ptr + output_offsets,
        grad_gate,
        mask=mask,
    )
    tl.store(
        grad_pair_ptr + output_offsets + INTERMEDIATE_SIZE,
        grad_up,
        mask=mask,
    )


@triton.jit
def _router_logits_backward_kernel(
    grad_output_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    score_mask_ptr,
    scores_ptr,
    grad_router_logits_ptr,
    HIDDEN_SIZE: tl.constexpr,
    N_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    token = tl.program_id(0)

    h_offsets = tl.arange(0, BLOCK_H)
    grad_output = tl.load(
        grad_output_ptr + token * HIDDEN_SIZE + h_offsets,
        mask=h_offsets < HIDDEN_SIZE,
        other=0.0,
    ).to(tl.float32)

    grad_norm_sq = tl.sum(grad_output * grad_output, axis=0)
    grad_topk = grad_norm_sq * (1.0 / TOP_K)

    k_offsets = tl.arange(0, TOP_K)
    topk_base = token * TOP_K
    topk_weights = tl.load(
        topk_weights_ptr + topk_base + k_offsets
    ).to(tl.float32)

    denominator = tl.sum(topk_weights, axis=0) + 1.0e-20
    weighted_grad_sum = tl.sum(
        grad_topk * topk_weights,
        axis=0,
    )
    sum_grad = weighted_grad_sum / denominator
    grad_before_norm = (
        grad_topk - sum_grad
    ) / denominator

    index0 = tl.load(topk_indices_ptr + topk_base + 0).to(tl.int32)
    index1 = tl.load(topk_indices_ptr + topk_base + 1).to(tl.int32)
    index2 = tl.load(topk_indices_ptr + topk_base + 2).to(tl.int32)
    index3 = tl.load(topk_indices_ptr + topk_base + 3).to(tl.int32)
    index4 = tl.load(topk_indices_ptr + topk_base + 4).to(tl.int32)
    index5 = tl.load(topk_indices_ptr + topk_base + 5).to(tl.int32)
    index6 = tl.load(topk_indices_ptr + topk_base + 6).to(tl.int32)
    index7 = tl.load(topk_indices_ptr + topk_base + 7).to(tl.int32)

    expert_offsets = tl.arange(0, BLOCK_E)
    expert_mask = expert_offsets < N_EXPERTS

    sparse_grad = tl.where(
        expert_offsets == index0, grad_before_norm, 0.0
    )
    sparse_grad += tl.where(
        expert_offsets == index1, grad_before_norm, 0.0
    )
    sparse_grad += tl.where(
        expert_offsets == index2, grad_before_norm, 0.0
    )
    sparse_grad += tl.where(
        expert_offsets == index3, grad_before_norm, 0.0
    )
    sparse_grad += tl.where(
        expert_offsets == index4, grad_before_norm, 0.0
    )
    sparse_grad += tl.where(
        expert_offsets == index5, grad_before_norm, 0.0
    )
    sparse_grad += tl.where(
        expert_offsets == index6, grad_before_norm, 0.0
    )
    sparse_grad += tl.where(
        expert_offsets == index7, grad_before_norm, 0.0
    )

    linear_offsets = token * N_EXPERTS + expert_offsets

    selection_mask = tl.load(
        score_mask_ptr + linear_offsets,
        mask=expert_mask,
        other=0.0,
    ).to(tl.float32)
    score = tl.load(
        scores_ptr + linear_offsets,
        mask=expert_mask,
        other=0.0,
    ).to(tl.float32)

    grad_router_logits = (
        sparse_grad
        * selection_mask
        * score
        * (1.0 - score)
    )

    tl.store(
        grad_router_logits_ptr + linear_offsets,
        grad_router_logits.to(tl.bfloat16),
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
    hidden_size = 4096
    intermediate_size = 1408
    pair_size = 2816
    n_routed_experts = 128
    num_experts_per_tok = 8

    grad_shared_activated = torch.matmul(
        grad_output,
        shared_expert_down_weight,
    )

    grad_shared_expert_down_weight = torch.matmul(
        grad_output.t(),
        shared_activated,
    )

    grad_shared_pair = torch.empty(
        (batch_seq_len, pair_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )

    _shared_swiglu_backward_paired_kernel[
        (
            batch_seq_len,
            triton.cdiv(intermediate_size, 256),
        )
    ](
        grad_shared_activated,
        shared_gate_output,
        shared_up_output,
        grad_shared_pair,
        INTERMEDIATE_SIZE=intermediate_size,
        PAIR_SIZE=pair_size,
        BLOCK_SIZE=256,
        num_warps=4,
    )

    grad_shared_gate_output = grad_shared_pair[:, :intermediate_size]
    grad_shared_up_output = grad_shared_pair[:, intermediate_size:]

    grad_shared_expert_pair_weight = torch.matmul(
        grad_shared_pair.t(),
        hidden_states,
    )
    grad_shared_expert_gate_weight = (
        grad_shared_expert_pair_weight[:intermediate_size]
    )
    grad_shared_expert_up_weight = (
        grad_shared_expert_pair_weight[intermediate_size:]
    )

    grad_hidden_states = torch.matmul(
        grad_shared_up_output,
        shared_expert_up_weight,
    )

    torch.addmm(
        grad_hidden_states,
        grad_shared_gate_output,
        shared_expert_gate_weight,
        beta=1,
        alpha=1,
        out=grad_hidden_states,
    )

    grad_router_logits_bf16 = torch.empty(
        (batch_seq_len, n_routed_experts),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )

    _router_logits_backward_kernel[(batch_seq_len,)](
        grad_output,
        topk_indices,
        topk_weights,
        score_mask,
        scores,
        grad_router_logits_bf16,
        HIDDEN_SIZE=hidden_size,
        N_EXPERTS=n_routed_experts,
        TOP_K=num_experts_per_tok,
        BLOCK_H=4096,
        BLOCK_E=128,
        num_warps=8,
    )

    grad_router_weight = torch.matmul(
        grad_router_logits_bf16.t(),
        hidden_states,
    ).to(torch.float32)

    torch.addmm(
        grad_hidden_states,
        grad_router_logits_bf16,
        router_weight,
        beta=1,
        alpha=1,
        out=grad_hidden_states,
    )

    return (
        grad_hidden_states,
        grad_router_weight,
        grad_shared_expert_gate_weight,
        grad_shared_expert_up_weight,
        grad_shared_expert_down_weight,
    )