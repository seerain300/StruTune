# task: 012_moe_expert_batched_execution_with_capacity_factor
# bench: SOL-L2 | batch: formal2_solL2_20260916
# final eval (official evaluator, full workloads): valid=True pass=16/16 geomean=1.440x
# feedback best (5-workload sample during search): 1.454x
# torch fallback audit: C·待消融 (bmm×3)
# tokens: 1,767,233

import torch
import triton
import triton.language as tl


@triton.jit
def _build_slots_kernel(
    selected_ptr,
    counters_ptr,
    slot_token_ptr,
    assignment_slot_ptr,
    capacity,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)

    tl.store(counters_ptr + offs, 0, mask=offs < E)
    tl.debug_barrier()

    mask = offs < N
    expert = tl.load(selected_ptr + offs, mask=mask, other=0).to(tl.int32)
    position = tl.atomic_add(counters_ptr + expert, 1, mask=mask)
    admitted = mask & (position < capacity)

    slot = expert * capacity + position
    token = offs // K

    tl.store(slot_token_ptr + slot, token, mask=admitted)
    tl.store(
        assignment_slot_ptr + offs,
        tl.where(admitted, slot, -1),
        mask=mask,
    )


@triton.jit
def _gather_expert_inputs_kernel(
    hidden_ptr,
    counters_ptr,
    slot_token_ptr,
    expert_input_ptr,
    capacity,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    expert = tl.program_id(0)
    block_m = tl.program_id(1)
    block_n = tl.program_id(2)

    rows = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = block_n * BLOCK_N + tl.arange(0, BLOCK_N)

    count = tl.load(counters_ptr + expert)
    valid_rows = (rows < capacity) & (rows < count)

    slots = expert * capacity + rows
    token_ids = tl.load(
        slot_token_ptr + slots,
        mask=valid_rows,
        other=0,
    ).to(tl.int32)

    values = tl.load(
        hidden_ptr + token_ids[:, None] * H + cols[None, :],
        mask=valid_rows[:, None],
        other=0.0,
    )

    output_offsets = slots[:, None] * H + cols[None, :]
    tl.store(
        expert_input_ptr + output_offsets,
        values,
        mask=valid_rows[:, None],
    )


@triton.jit
def _swiglu_inplace_kernel(
    gate_ptr,
    up_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    activated = gate * tl.sigmoid(gate) * up
    tl.store(gate_ptr + offsets, activated.to(tl.bfloat16), mask=mask)


@triton.jit
def _gather_weighted_output_kernel(
    expert_output_ptr,
    assignment_slot_ptr,
    routing_ptr,
    output_ptr,
    H: tl.constexpr,
    K: tl.constexpr,
    T: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tokens = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_tokens = tokens < T

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    assignment_base = tokens * K

    for k in range(K):
        assignment = assignment_base + k
        slot = tl.load(
            assignment_slot_ptr + assignment,
            mask=valid_tokens,
            other=-1,
        ).to(tl.int32)
        valid = valid_tokens & (slot >= 0)

        weight = tl.load(
            routing_ptr + assignment,
            mask=valid,
            other=0.0,
        ).to(tl.float32)

        projected = tl.load(
            expert_output_ptr + slot[:, None] * H + cols[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)

        weighted = (projected * weight[:, None]).to(tl.bfloat16)
        accumulator += weighted.to(tl.float32)

    tl.store(
        output_ptr + tokens[:, None] * H + cols[None, :],
        accumulator.to(tl.bfloat16),
        mask=valid_tokens[:, None],
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
    experts_per_token = selected_experts.shape[1]

    capacity = max(
        (num_tokens * experts_per_token * 5) // (num_experts * 4),
        1,
    )
    num_assignments = num_tokens * experts_per_token
    num_slots = num_experts * capacity
    device = hidden_states.device

    metadata = torch.empty(
        num_experts + num_slots + num_assignments,
        dtype=torch.int32,
        device=device,
    )
    counters = metadata[:num_experts]
    slot_tokens = metadata[num_experts:num_experts + num_slots]
    assignment_slots = metadata[num_experts + num_slots:]

    routing_block = triton.next_power_of_2(
        max(num_assignments, num_experts)
    )
    _build_slots_kernel[(1,)](
        selected_experts,
        counters,
        slot_tokens,
        assignment_slots,
        capacity,
        N=num_assignments,
        K=experts_per_token,
        E=num_experts,
        BLOCK=routing_block,
        num_warps=8,
    )

    expert_input_elements = num_slots * hidden_size
    projection_elements = num_slots * intermediate_size
    workspace = torch.empty(
        expert_input_elements + 2 * projection_elements,
        dtype=hidden_states.dtype,
        device=device,
    )

    expert_inputs = workspace[:expert_input_elements].view(
        num_experts,
        capacity,
        hidden_size,
    )
    projection_outputs = workspace[
        expert_input_elements:
    ].view(
        2,
        num_experts,
        capacity,
        intermediate_size,
    )

    gather_block_m = 16
    gather_block_n = 1024
    _gather_expert_inputs_kernel[
        (
            num_experts,
            triton.cdiv(capacity, gather_block_m),
            hidden_size // gather_block_n,
        )
    ](
        hidden_states,
        counters,
        slot_tokens,
        expert_inputs,
        capacity,
        H=hidden_size,
        BLOCK_M=gather_block_m,
        BLOCK_N=gather_block_n,
        num_warps=8,
    )

    gate_output = projection_outputs[0]
    up_output = projection_outputs[1]

    torch.bmm(
        expert_inputs,
        expert_gate_weights,
        out=gate_output,
    )
    torch.bmm(
        expert_inputs,
        expert_up_weights,
        out=up_output,
    )

    activation_elements = num_slots * intermediate_size
    activation_block = 1024
    _swiglu_inplace_kernel[
        (triton.cdiv(activation_elements, activation_block),)
    ](
        gate_output,
        up_output,
        N=activation_elements,
        BLOCK=activation_block,
        num_warps=4,
    )

    torch.bmm(
        gate_output,
        expert_down_weights,
        out=expert_inputs,
    )
    expert_outputs = expert_inputs

    output = torch.empty(
        (num_tokens, hidden_size),
        dtype=hidden_states.dtype,
        device=device,
    )

    if num_tokens <= 16:
        output_block_m = 1
        output_block_n = 256
        output_warps = 4
    elif num_tokens < 512:
        output_block_m = 4
        output_block_n = 256
        output_warps = 8
    else:
        output_block_m = 1
        output_block_n = 1024
        output_warps = 8

    _gather_weighted_output_kernel[
        (
            triton.cdiv(num_tokens, output_block_m),
            hidden_size // output_block_n,
        )
    ](
        expert_outputs,
        assignment_slots,
        routing_weights,
        output,
        H=hidden_size,
        K=experts_per_token,
        T=num_tokens,
        BLOCK_M=output_block_m,
        BLOCK_N=output_block_n,
        num_warps=output_warps,
    )

    return output