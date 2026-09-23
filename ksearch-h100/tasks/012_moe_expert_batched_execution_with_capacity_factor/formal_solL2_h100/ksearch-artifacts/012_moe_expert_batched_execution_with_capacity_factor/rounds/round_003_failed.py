# solution=GPT-5.6-Sol_012_moe_expert_batched_execution_with_capacity_factor_triton_optimized_r3 score=-1.0 passed=False
import torch
import triton
import triton.language as tl


@triton.jit
def _pack_expert_inputs_kernel(
    hidden_states,
    sorted_assignment_ids,
    expert_starts,
    expert_counts,
    expert_inputs,
    HIDDEN_SIZE: tl.constexpr,
    TOP_K: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    expert_id = tl.program_id(0)
    row_block = tl.program_id(1)
    hidden_block = tl.program_id(2)

    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = hidden_block * BLOCK_H + tl.arange(0, BLOCK_H)

    expert_start = tl.load(expert_starts + expert_id)
    expert_count = tl.load(expert_counts + expert_id)
    row_mask = (rows < CAPACITY) & (rows < expert_count)

    assignment_ids = tl.load(
        sorted_assignment_ids + expert_start + rows,
        mask=row_mask,
        other=0,
    )
    token_ids = assignment_ids // TOP_K

    values = tl.load(
        hidden_states
        + token_ids[:, None].to(tl.int64) * HIDDEN_SIZE
        + cols[None, :],
        mask=row_mask[:, None] & (cols[None, :] < HIDDEN_SIZE),
        other=0.0,
    )

    output_offsets = (
        expert_id.to(tl.int64) * CAPACITY * HIDDEN_SIZE
        + rows[:, None].to(tl.int64) * HIDDEN_SIZE
        + cols[None, :]
    )
    tl.store(
        expert_inputs + output_offsets,
        values,
        mask=row_mask[:, None] & (cols[None, :] < HIDDEN_SIZE),
    )


@triton.jit
def _swiglu_inplace_kernel(
    gate_output,
    up_output,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    gate = tl.load(gate_output + offsets, mask=mask, other=0.0)
    up = tl.load(up_output + offsets, mask=mask, other=0.0)
    activated = gate * tl.sigmoid(gate) * up

    tl.store(gate_output + offsets, activated, mask=mask)


@triton.jit
def _scatter_expert_outputs_kernel(
    expert_outputs,
    sorted_assignment_ids,
    expert_starts,
    expert_counts,
    routing_weights,
    output_accumulator,
    HIDDEN_SIZE: tl.constexpr,
    TOP_K: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    expert_id = tl.program_id(0)
    row_block = tl.program_id(1)
    hidden_block = tl.program_id(2)

    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = hidden_block * BLOCK_H + tl.arange(0, BLOCK_H)

    expert_start = tl.load(expert_starts + expert_id)
    expert_count = tl.load(expert_counts + expert_id)
    row_mask = (rows < CAPACITY) & (rows < expert_count)

    assignment_ids = tl.load(
        sorted_assignment_ids + expert_start + rows,
        mask=row_mask,
        other=0,
    )
    token_ids = assignment_ids // TOP_K

    expert_offsets = (
        expert_id.to(tl.int64) * CAPACITY * HIDDEN_SIZE
        + rows[:, None].to(tl.int64) * HIDDEN_SIZE
        + cols[None, :]
    )
    values = tl.load(
        expert_outputs + expert_offsets,
        mask=row_mask[:, None] & (cols[None, :] < HIDDEN_SIZE),
        other=0.0,
    )

    route_weights = tl.load(
        routing_weights + assignment_ids,
        mask=row_mask,
        other=0.0,
    )
    weighted = (values * route_weights[:, None]).to(tl.bfloat16).to(tl.float32)

    tl.atomic_add(
        output_accumulator
        + token_ids[:, None].to(tl.int64) * HIDDEN_SIZE
        + cols[None, :],
        weighted,
        mask=row_mask[:, None] & (cols[None, :] < HIDDEN_SIZE),
    )


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_gate_weights: torch.Tensor,
    expert_up_weights: torch.Tensor,
    expert_down_weights: torch.Tensor,
):
    num_tokens, hidden_size = hidden_states.shape
    num_experts, _, intermediate_size = expert_gate_weights.shape
    top_k = selected_experts.shape[1]

    capacity = max(
        int((num_tokens * top_k / num_experts) * 1.25),
        1,
    )

    flat_experts = selected_experts.reshape(-1)
    sorted_experts, sorted_assignment_ids = torch.sort(
        flat_experts,
        stable=True,
    )

    expert_counts = torch.bincount(
        sorted_experts,
        minlength=num_experts,
    )
    expert_starts = torch.empty_like(expert_counts)
    expert_starts[0] = 0
    expert_starts[1:] = expert_counts[:-1].cumsum(0)

    expert_inputs = torch.empty(
        (num_experts, capacity, hidden_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    block_m = 16
    block_h = 256
    row_blocks = triton.cdiv(capacity, block_m)

    _pack_expert_inputs_kernel[
        (
            num_experts,
            row_blocks,
            triton.cdiv(hidden_size, block_h),
        )
    ](
        hidden_states,
        sorted_assignment_ids,
        expert_starts,
        expert_counts,
        expert_inputs,
        HIDDEN_SIZE=hidden_size,
        TOP_K=top_k,
        CAPACITY=capacity,
        BLOCK_M=block_m,
        BLOCK_H=block_h,
        num_warps=8,
    )

    gate_output = torch.bmm(expert_inputs, expert_gate_weights)
    up_output = torch.bmm(expert_inputs, expert_up_weights)

    activation_elements = num_experts * capacity * intermediate_size
    activation_block = 1024
    _swiglu_inplace_kernel[
        (triton.cdiv(activation_elements, activation_block),)
    ](
        gate_output,
        up_output,
        activation_elements,
        BLOCK_SIZE=activation_block,
        num_warps=8,
    )

    expert_outputs = torch.bmm(gate_output, expert_down_weights)

    output_accumulator = torch.zeros(
        (num_tokens, hidden_size),
        dtype=torch.float32,
        device=hidden_states.device,
    )

    _scatter_expert_outputs_kernel[
        (
            num_experts,
            row_blocks,
            triton.cdiv(hidden_size, block_h),
        )
    ](
        expert_outputs,
        sorted_assignment_ids,
        expert_starts,
        expert_counts,
        routing_weights.reshape(-1),
        output_accumulator,
        HIDDEN_SIZE=hidden_size,
        TOP_K=top_k,
        CAPACITY=capacity,
        BLOCK_M=block_m,
        BLOCK_H=block_h,
        num_warps=8,
    )

    return output_accumulator.to(torch.bfloat16)