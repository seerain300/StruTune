# solution=GPT-5.6-Sol_080_moe_complete_layer_with_shared_expert_backward_triton_optimized_r65 score=-1.0 passed=False
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

    grad_hidden_states = torch.empty_like(hidden_states)

    if batch_seq_len <= 512:
        block_m = 16
        block_n = 128
        block_k = 32
        group_m = 8
        input_num_warps = 4
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
        block_k = 32
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