# solution=GPT-5.6-Sol_012_moe_expert_batched_execution_with_capacity_factor_triton_optimized_r1 score=-1.0 passed=False
import torch
import triton
import triton.language as tl


_NUM_EXPERTS = 160
_HIDDEN_SIZE = 6144
_INTERMEDIATE_SIZE = 2560
_TOP_K = 8


@triton.jit
def _build_slots_kernel(
    selected_experts_ptr,
    slot_tokens_ptr,
    assignment_slots_ptr,
    total_assignments,
    capacity: tl.constexpr,
    num_experts: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    expert = tl.program_id(0)
    block_offsets = tl.arange(0, BLOCK_SIZE)

    for slot_start in range(0, capacity, BLOCK_SIZE):
        positions = slot_start + block_offsets
        tl.store(
            slot_tokens_ptr + expert * capacity + positions,
            -1,
            mask=positions < capacity,
        )

    assignment_start = 0
    expert_count = 0

    while (assignment_start < total_assignments) & (expert_count < capacity):
        assignments = assignment_start + block_offsets
        assignment_mask = assignments < total_assignments

        selected = tl.load(
            selected_experts_ptr + assignments,
            mask=assignment_mask,
            other=-1,
        )
        matches = assignment_mask & (selected == expert)
        ranks = expert_count + tl.cumsum(matches.to(tl.int32), axis=0) - 1
        admitted = matches & (ranks < capacity)

        slots = expert * capacity + ranks
        tokens = assignments // top_k

        tl.store(
            slot_tokens_ptr + slots,
            tokens,
            mask=admitted,
        )
        tl.store(
            assignment_slots_ptr + assignments,
            slots,
            mask=admitted,
        )

        expert_count += tl.sum(matches.to(tl.int32), axis=0)
        assignment_start += BLOCK_SIZE


@triton.jit
def _gather_dual_projection_kernel(
    hidden_states_ptr,
    slot_tokens_ptr,
    expert_gate_weights_ptr,
    expert_up_weights_ptr,
    gate_out_ptr,
    up_out_ptr,
    total_rows,
    capacity,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    n_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    row_mask = row < total_rows
    n_mask = n_offsets < intermediate_size

    token = tl.load(
        slot_tokens_ptr + row,
        mask=row_mask,
        other=-1,
    )
    valid_row = row_mask & (token >= 0)

    expert = row // capacity
    expert_weight_base = expert * hidden_size * intermediate_size

    gate_acc = tl.zeros((1, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((1, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, hidden_size, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < hidden_size

        hidden = tl.load(
            hidden_states_ptr
            + token * hidden_size
            + k_offsets,
            mask=valid_row & k_mask,
            other=0.0,
        )

        gate_weights = tl.load(
            expert_gate_weights_ptr
            + expert_weight_base
            + k_offsets[:, None] * intermediate_size
            + n_offsets[None, :],
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )

        up_weights = tl.load(
            expert_up_weights_ptr
            + expert_weight_base
            + k_offsets[:, None] * intermediate_size
            + n_offsets[None, :],
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )

        gate_acc += tl.dot(hidden[None, :], gate_weights)
        up_acc += tl.dot(hidden[None, :], up_weights)

    output_offsets = row * intermediate_size + n_offsets

    tl.store(
        gate_out_ptr + output_offsets,
        gate_acc[0, :],
        mask=row_mask & n_mask,
    )
    tl.store(
        up_out_ptr + output_offsets,
        up_acc[0, :],
        mask=row_mask & n_mask,
    )


@triton.jit
def _swiglu_kernel(
    gate_ptr,
    up_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    activated = (gate * tl.sigmoid(gate)) * up

    tl.store(gate_ptr + offsets, activated, mask=mask)


@triton.jit
def _gather_weighted_output_kernel(
    expert_outputs_ptr,
    assignment_slots_ptr,
    routing_weights_ptr,
    result_ptr,
    num_tokens,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    tokens = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    columns = tl.program_id(1) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)

    token_mask = tokens < num_tokens
    column_mask = columns < hidden_size
    accumulator = tl.zeros((BLOCK_ROWS, BLOCK_COLS), dtype=tl.float32)

    for k in range(0, top_k):
        assignments = tokens * top_k + k
        slots = tl.load(
            assignment_slots_ptr + assignments,
            mask=token_mask,
            other=-1,
        )
        valid = token_mask & (slots >= 0)
        weights = tl.load(
            routing_weights_ptr + assignments,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        values = tl.load(
            expert_outputs_ptr
            + slots[:, None].to(tl.int64) * hidden_size
            + columns[None, :],
            mask=valid[:, None] & column_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        accumulator += values * weights[:, None]

    tl.store(
        result_ptr
        + tokens[:, None].to(tl.int64) * hidden_size
        + columns[None, :],
        accumulator,
        mask=token_mask[:, None] & column_mask[None, :],
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
    num_tokens = hidden_states.shape[0]
    capacity = max(num_tokens // 16, 1)
    device = hidden_states.device

    total_rows = _NUM_EXPERTS * capacity
    total_assignments = num_tokens * _TOP_K

    slot_tokens = torch.empty(
        total_rows,
        dtype=torch.int32,
        device=device,
    )
    assignment_slots = torch.full(
        (total_assignments,),
        -1,
        dtype=torch.int32,
        device=device,
    )

    _build_slots_kernel[(_NUM_EXPERTS,)](
        selected_experts,
        slot_tokens,
        assignment_slots,
        total_assignments,
        capacity=capacity,
        num_experts=_NUM_EXPERTS,
        top_k=_TOP_K,
        BLOCK_SIZE=256,
        num_warps=4,
    )

    gate_out = torch.empty(
        (_NUM_EXPERTS, capacity, _INTERMEDIATE_SIZE),
        dtype=hidden_states.dtype,
        device=device,
    )
    up_out = torch.empty_like(gate_out)

    _gather_dual_projection_kernel[
        (
            total_rows,
            triton.cdiv(_INTERMEDIATE_SIZE, 256),
        )
    ](
        hidden_states,
        slot_tokens,
        expert_gate_weights,
        expert_up_weights,
        gate_out,
        up_out,
        total_rows,
        capacity,
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=_INTERMEDIATE_SIZE,
        BLOCK_N=256,
        BLOCK_K=256,
        num_warps=8,
    )

    num_activated_elements = total_rows * _INTERMEDIATE_SIZE
    _swiglu_kernel[
        (triton.cdiv(num_activated_elements, 1024),)
    ](
        gate_out,
        up_out,
        num_activated_elements,
        BLOCK_SIZE=1024,
        num_warps=8,
    )

    expert_outputs = torch.bmm(gate_out, expert_down_weights)

    result = torch.empty_like(hidden_states)
    _gather_weighted_output_kernel[
        (
            triton.cdiv(num_tokens, 2),
            triton.cdiv(_HIDDEN_SIZE, 256),
        )
    ](
        expert_outputs,
        assignment_slots,
        routing_weights,
        result,
        num_tokens,
        hidden_size=_HIDDEN_SIZE,
        top_k=_TOP_K,
        BLOCK_ROWS=2,
        BLOCK_COLS=256,
        num_warps=8,
    )

    return result