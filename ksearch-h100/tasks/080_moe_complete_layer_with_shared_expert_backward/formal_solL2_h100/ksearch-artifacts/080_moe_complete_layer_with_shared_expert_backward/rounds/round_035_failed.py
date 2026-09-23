# solution=GPT-5.6-Sol_080_moe_complete_layer_with_shared_expert_backward_triton_optimized_r35 score=-1.0 passed=False
import torch
import triton
import triton.language as tl


_INTERMEDIATE_SIZE = 1408
_N_ROUTED_EXPERTS = 128
_HIDDEN_SIZE = 4096


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

    grad_gate_silu = (grad_activated * up).to(tl.bfloat16)
    silu_gate = (gate * sigmoid_gate).to(tl.bfloat16)
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

    _shared_swiglu_backward_kernel[
        (
            batch_seq_len,
            triton.cdiv(_INTERMEDIATE_SIZE, 256),
        )
    ](
        grad_shared_activated,
        shared_gate_output,
        shared_up_output,
        grad_gate_up,
        _INTERMEDIATE_SIZE,
        2 * _INTERMEDIATE_SIZE,
        BLOCK_SIZE=256,
        num_warps=4,
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

    grad_shared_gate_output = grad_gate_up[:, :_INTERMEDIATE_SIZE]
    grad_shared_up_output = grad_gate_up[:, _INTERMEDIATE_SIZE:]

    grad_hidden_states = torch.mm(
        grad_shared_up_output,
        shared_expert_up_weight,
    )
    grad_hidden_states.addmm_(
        grad_shared_gate_output,
        shared_expert_gate_weight,
    )

    grad_router_weight = torch.zeros(
        (_N_ROUTED_EXPERTS, _HIDDEN_SIZE),
        dtype=torch.float32,
        device=hidden_states.device,
    )

    return (
        grad_hidden_states,
        grad_router_weight,
        grad_shared_expert_gate_weight,
        grad_shared_expert_up_weight,
        grad_shared_expert_down_weight,
    )