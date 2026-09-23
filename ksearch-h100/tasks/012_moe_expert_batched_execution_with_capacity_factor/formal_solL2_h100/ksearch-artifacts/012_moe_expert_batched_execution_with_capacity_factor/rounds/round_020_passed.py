# solution=GPT-5.6-Sol_012_moe_expert_batched_execution_with_capacity_factor_triton_optimized_r20 score=1.476660694122812 passed=True
import torch
import triton
import triton.language as tl


@triton.jit
def _build_admission_ranks_kernel(
    sorted_experts,
    sorted_indices,
    expert_starts,
    packed_metadata,
    NUM_ASSIGNMENTS: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    positions = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = positions < NUM_ASSIGNMENTS

    experts = tl.load(sorted_experts + positions, mask=mask, other=0)
    assignments = tl.load(sorted_indices + positions, mask=mask, other=0)
    starts = tl.load(expert_starts + experts, mask=mask, other=0)

    ranks = positions - starts
    admitted = ranks < CAPACITY
    packed = (experts * CAPACITY + ranks).to(tl.int32)
    packed = tl.where(admitted, packed, -1)

    tl.store(
        packed_metadata + assignments,
        packed,
        mask=mask,
    )


@triton.jit
def _pack_expert_inputs_kernel(
    hidden_states,
    sorted_indices,
    expert_starts,
    expert_counts,
    expert_inputs,
    hidden_stride_t,
    hidden_stride_h,
    input_stride_e,
    input_stride_c,
    input_stride_h,
    CAPACITY: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    expert = tl.program_id(0)
    row_block = tl.program_id(1)
    col_block = tl.program_id(2)

    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)

    start = tl.load(expert_starts + expert)
    count = tl.load(expert_counts + expert)

    row_mask = rows < CAPACITY
    admitted = row_mask & (rows < count)
    col_mask = cols < HIDDEN_SIZE

    assignments = tl.load(
        sorted_indices + start + rows,
        mask=admitted,
        other=0,
    )
    tokens = assignments // TOP_K

    values = tl.load(
        hidden_states
        + tokens[:, None] * hidden_stride_t
        + cols[None, :] * hidden_stride_h,
        mask=admitted[:, None] & col_mask[None, :],
        other=0.0,
    )

    tl.store(
        expert_inputs
        + expert * input_stride_e
        + rows[:, None] * input_stride_c
        + cols[None, :] * input_stride_h,
        values,
        mask=row_mask[:, None] & col_mask[None, :],
    )


@triton.jit
def _swiglu_kernel(
    gate,
    up,
    NUM_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUM_ELEMENTS

    gate_value = tl.load(gate + offsets, mask=mask, other=0.0).to(tl.float32)
    up_value = tl.load(up + offsets, mask=mask, other=0.0)

    sigmoid_value = 1.0 / (1.0 + tl.exp(-gate_value))
    silu_value = (gate_value * sigmoid_value).to(tl.bfloat16)
    activated = silu_value * up_value

    tl.store(gate + offsets, activated, mask=mask)


@triton.jit
def _token_major_reduce_kernel(
    expert_outputs,
    routing_weights,
    packed_metadata,
    output,
    expert_output_stride_c,
    expert_output_stride_h,
    routing_stride_t,
    routing_stride_k,
    output_stride_t,
    output_stride_h,
    HIDDEN_SIZE: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token = tl.program_id(0)
    col_block = tl.program_id(1)

    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < HIDDEN_SIZE
    result = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for route in range(TOP_K):
        assignment = token * TOP_K + route
        packed = tl.load(packed_metadata + assignment)
        admitted = packed >= 0
        packed_safe = tl.maximum(packed, 0)

        routing_weight = tl.load(
            routing_weights
            + token * routing_stride_t
            + route * routing_stride_k
        )

        value = tl.load(
            expert_outputs
            + packed_safe * expert_output_stride_c
            + cols * expert_output_stride_h,
            mask=admitted & col_mask,
            other=0.0,
        )

        contribution = (value * routing_weight).to(tl.bfloat16)
        result += contribution.to(tl.float32)

    tl.store(
        output
        + token * output_stride_t
        + cols * output_stride_h,
        result.to(tl.bfloat16),
        mask=col_mask,
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
    num_assignments = num_tokens * top_k

    capacity = max(
        int((num_assignments / num_experts) * 1.25),
        1,
    )

    flat_experts = selected_experts.reshape(-1)
    sorted_experts, sorted_indices = flat_experts.sort(stable=True)

    expert_counts = torch.bincount(
        sorted_experts,
        minlength=num_experts,
    )

    expert_starts = torch.empty_like(expert_counts)
    expert_starts[0] = 0
    expert_starts[1:] = expert_counts[:-1].cumsum(0)

    packed_metadata = torch.empty(
        num_assignments,
        dtype=torch.int32,
        device=hidden_states.device,
    )

    rank_block = 1024
    _build_admission_ranks_kernel[
        (triton.cdiv(num_assignments, rank_block),)
    ](
        sorted_experts,
        sorted_indices,
        expert_starts,
        packed_metadata,
        NUM_ASSIGNMENTS=num_assignments,
        CAPACITY=capacity,
        BLOCK_SIZE=rank_block,
        num_warps=4,
    )

    expert_inputs = torch.empty(
        (num_experts, capacity, hidden_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    pack_block_m = 64
    pack_block_n = 256
    pack_grid = (
        num_experts,
        triton.cdiv(capacity, pack_block_m),
        triton.cdiv(hidden_size, pack_block_n),
    )

    _pack_expert_inputs_kernel[pack_grid](
        hidden_states,
        sorted_indices,
        expert_starts,
        expert_counts,
        expert_inputs,
        hidden_states.stride(0),
        hidden_states.stride(1),
        expert_inputs.stride(0),
        expert_inputs.stride(1),
        expert_inputs.stride(2),
        CAPACITY=capacity,
        HIDDEN_SIZE=hidden_size,
        TOP_K=top_k,
        BLOCK_M=pack_block_m,
        BLOCK_N=pack_block_n,
        num_warps=8,
    )

    gate_out = torch.bmm(expert_inputs, expert_gate_weights)
    up_out = torch.bmm(expert_inputs, expert_up_weights)

    activation_elements = num_experts * capacity * intermediate_size
    activation_block = 8192

    _swiglu_kernel[
        (triton.cdiv(activation_elements, activation_block),)
    ](
        gate_out,
        up_out,
        NUM_ELEMENTS=activation_elements,
        BLOCK_SIZE=activation_block,
        num_warps=8,
    )

    expert_outputs = torch.bmm(gate_out, expert_down_weights)

    output = torch.empty(
        (num_tokens, hidden_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    reduce_block_n = 128
    reduce_grid = (
        num_tokens,
        triton.cdiv(hidden_size, reduce_block_n),
    )

    _token_major_reduce_kernel[reduce_grid](
        expert_outputs,
        routing_weights,
        packed_metadata,
        output,
        expert_outputs.stride(1),
        expert_outputs.stride(2),
        routing_weights.stride(0),
        routing_weights.stride(1),
        output.stride(0),
        output.stride(1),
        HIDDEN_SIZE=hidden_size,
        TOP_K=top_k,
        BLOCK_N=reduce_block_n,
        num_warps=4,
    )

    return output