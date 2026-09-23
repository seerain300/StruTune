# solution=GPT-5.6-Sol_080_moe_complete_layer_with_shared_expert_backward_triton_optimized_r1 score=-1.0 passed=False
import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_backward_kernel(
    grad_activated_ptr,
    gate_ptr,
    up_ptr,
    grad_gate_ptr,
    grad_up_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_activated = tl.load(
        grad_activated_ptr + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sigmoid_gate = tl.sigmoid(gate)
    silu_gate = (gate * sigmoid_gate).to(tl.bfloat16)
    grad_gate_silu = (grad_activated * up).to(tl.bfloat16)

    grad_up = grad_activated * silu_gate.to(tl.float32)
    silu_derivative = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
    grad_gate = grad_gate_silu.to(tl.float32) * silu_derivative

    tl.store(grad_gate_ptr + offsets, grad_gate, mask=mask)
    tl.store(grad_up_ptr + offsets, grad_up, mask=mask)


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
    intermediate_size = shared_gate_output.shape[1]

    grad_shared_activated = torch.mm(
        grad_output, shared_expert_down_weight
    )

    grad_shared_expert_down_weight = torch.mm(
        grad_output.transpose(0, 1), shared_activated
    )

    grad_shared_gate_output = torch.empty_like(shared_gate_output)
    grad_shared_up_output = torch.empty_like(shared_up_output)

    n_elements = batch_seq_len * intermediate_size
    _swiglu_backward_kernel[
        (triton.cdiv(n_elements, 256),)
    ](
        grad_shared_activated,
        shared_gate_output,
        shared_up_output,
        grad_shared_gate_output,
        grad_shared_up_output,
        n_elements,
        BLOCK_SIZE=256,
        num_warps=4,
    )

    grad_shared_expert_up_weight = torch.mm(
        grad_shared_up_output.transpose(0, 1), hidden_states
    )
    grad_shared_expert_gate_weight = torch.mm(
        grad_shared_gate_output.transpose(0, 1), hidden_states
    )

    grad_hidden_states = torch.mm(
        grad_shared_up_output, shared_expert_up_weight
    )
    grad_hidden_from_gate = torch.mm(
        grad_shared_gate_output, shared_expert_gate_weight
    )
    grad_hidden_states.add_(grad_hidden_from_gate)

    grad_router_weight = torch.zeros(
        router_weight.shape,
        dtype=torch.float32,
        device=router_weight.device,
    )

    return (
        grad_hidden_states,
        grad_router_weight,
        grad_shared_expert_gate_weight,
        grad_shared_expert_up_weight,
        grad_shared_expert_down_weight,
    )