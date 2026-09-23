import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: flatten selected_experts [num_tokens, num_experts_per_tok] into int32 flattened array
@triton.jit
def flatten_selected_exp_kernel(
    selected_experts_ptr,   # *int64, shape [num_tokens, num_experts_per_tok]
    flattened_exp_ptr,      # *int32, shape [E]
    num_tokens: tl.constexpr,
    num_experts_per_tok: tl.constexpr,
    E: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * 128 + tl.arange(0, 128)
    mask = offs < E
    # Load expert id: selected_experts[row, col] with row = offs // num_experts_per_tok
    row = offs // num_experts_per_tok
    col = offs % num_experts_per_tok
    # Compute pointer: each token row is contiguous in memory
    ptr = selected_experts_ptr + row * num_experts_per_tok + col
    exp = tl.load(ptr, mask=mask, other=0).to(tl.int32)
    tl.store(flattened_exp_ptr + offs, exp, mask=mask)


# Triton kernel: flatten routing_weights [num_tokens, num_experts_per_tok] into bfloat16 flattened array
@triton.jit
def flatten_weights_kernel(
    routing_weights_ptr,    # *bfloat16, shape [num_tokens, num_experts_per_tok]
    flattened_wt_ptr,       # *bfloat16, shape [E]
    num_tokens: tl.constexpr,
    num_experts_per_tok: tl.constexpr,
    E: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * 128 + tl.arange(0, 128)
    mask = offs < E
    row = offs // num_experts_per_tok
    col = offs % num_experts_per_tok
    ptr = routing_weights_ptr + row * num_experts_per_tok + col
    wt = tl.load(ptr, mask=mask, other=tl.zeros((), dtype=tl.bfloat16))
    tl.store(flattened_wt_ptr + offs, wt, mask=mask)


# Triton kernel: odd-even stable sort on flattened_exp (int32) and track sorted positions
# to_sort: input int32, sorted_exp: output int32, pos: output int32 (sorted positions), E: number of elements
@triton.jit
def odd_even_stable_sort_experts_by_id_kernel(
    to_sort_ptr,            # *int32
    sorted_exp_ptr,         # *int32
    pos_ptr,                # *int32
    E: tl.constexpr,
    T: tl.constexpr,        # number of iterations = E // 2 + (E % 2)
):
    # Perform odd-even sort: T phases
    for t in range(0, T):
        # Even phase: compare (0,1), (2,3), ...
        if (t % 2) == 0:
            i = 2 * tl.arange(0, 128)  # even indices
            mask = i + 1 < E
            a = tl.load(to_sort_ptr + i, mask=mask, other=tl.full((), 0, dtype=tl.int32))
            b = tl.load(to_sort_ptr + i + 1, mask=mask, other=tl.full((), 1, dtype=tl.int32))
            # compare
            swap = a > b
            minv = tl.where(swap, b, a)
            maxv = tl.where(swap, a, b)
            # write back
            tl.store(to_sort_ptr + i, minv, mask=mask)
            tl.store(to_sort_ptr + i + 1, maxv, mask=mask)
            # update positions accordingly
            a_pos = tl.load(pos_ptr + i, mask=mask, other=0)
            b_pos = tl.load(pos_ptr + i + 1, mask=mask, other=1)
            new_a_pos = tl.where(swap, b_pos, a_pos)
            new_b_pos = tl.where(swap, a_pos, b_pos)
            tl.store(pos_ptr + i, new_a_pos, mask=mask)
            tl.store(pos_ptr + i + 1, new_b_pos, mask=mask)
        # Odd phase: compare (1,2), (3,4), ...
        else:
            i = 2 * tl.arange(0, 128) + 1
            mask = i + 1 < E
            a = tl.load(to_sort_ptr + i, mask=mask, other=tl.full((), 0, dtype=tl.int32))
            b = tl.load(to_sort_ptr + i + 1, mask=mask, other=tl.full((), 1, dtype=tl.int32))
            swap = a > b
            minv = tl.where(swap, b, a)
            maxv = tl.where(swap, a, b)
            tl.store(to_sort_ptr + i, minv, mask=mask)
            tl.store(to_sort_ptr + i + 1, maxv, mask=mask)
            a_pos = tl.load(pos_ptr + i, mask=mask, other=0)
            b_pos = tl.load(pos_ptr + i + 1, mask=mask, other=1)
            new_a_pos = tl.where(swap, b_pos, a_pos)
            new_b_pos = tl.where(swap, a_pos, b_pos)
            tl.store(pos_ptr + i, new_a_pos, mask=mask)
            tl.store(pos_ptr + i + 1, new_b_pos, mask=mask)
    # After sorting, write sorted_exp = to_sort (sorted) and pos (stable order)
    offs = tl.arange(0, 128)
    mask = offs < E
    sorted_vals = tl.load(to_sort_ptr + offs, mask=mask, other=0)
    stable_pos = tl.load(pos_ptr + offs, mask=mask, other=0)
    tl.store(sorted_exp_ptr + offs, sorted_vals, mask=mask)
    tl.store(pos_ptr + offs, stable_pos, mask=mask)


# Triton kernel: per-expert counts via bincount on sorted_exp
@triton.jit
def bincount_experts_kernel(
    sorted_exp_ptr,      # *int32, [E]
    counts_ptr,          # *int32, [num_experts]
    E: tl.constexpr,
    num_experts: tl.constexpr,
):
    # Each program handles one expert id: counts[id] = number of occurrences
    pid = tl.program_id(axis=0)
    # Compute count by scanning the array and summing mask
    acc = 0
    # We implement a simple loop over E in chunks
    for i in range(0, E):
        val = tl.load(sorted_exp_ptr + i)
        is_exp = val == pid
        acc += is_exp.to(tl.int32)
    tl.store(counts_ptr + pid, acc)


# Triton kernel: cumsum of counts to get starts per expert
@triton.jit
def cumsum_starts_kernel(
    counts_ptr,          # *int32, [num_experts]
    starts_ptr,          # *int32, [num_experts]
    num_experts: tl.constexpr,
):
    # Sequential scan for starts[i] = sum_{j=0..i} counts[j]
    total = 0
    for i in range(0, num_experts):
        total += tl.load(counts_ptr + i)
        tl.store(starts_ptr + i, total)


# Triton kernel: compute within_pos and validity mask for each flattened index
@triton.jit
def compute_within_pos_valid_kernel(
    sorted_exp_ptr,          # *int32, [E]
    starts_ptr,              # *int32, [num_experts]
    capacity_scalar,         # int32
    valid_ptr,               # *int32, [E]
    E: tl.constexpr,
):
    offs = tl.arange(0, 128)
    mask = offs < E
    exp = tl.load(sorted_exp_ptr + offs, mask=mask, other=0)
    start = tl.load(starts_ptr + exp, mask=mask, other=0)
    within_pos = offs - start
    cond = within_pos < capacity_scalar
    tl.store(valid_ptr + offs, cond.to(tl.int32), mask=mask)


# Triton kernel: final weighted aggregation with hidden_states, expert weights, and routing weights.
# For each flattened index, load token, expert, weight, hidden_state, compute expert forward (gate, up, down, SiLU, multiply),
# then atomic_add into result_fp32[token, :] with the routing weight.
@triton.jit
def scatter_weighted_add_kernel(
    sorted_exp_ptr,            # *int32, [E]
    flattened_wt_ptr,          # *bfloat16, [E]
    hidden_states_ptr,         # *bfloat16, [num_tokens, hidden_size]
    expert_gate_w_ptr,         # *bfloat16, [num_experts, hidden_size, intermediate_size]
    expert_up_w_ptr,           # *bfloat16, [num_experts, hidden_size, intermediate_size]
    expert_down_w_ptr,         # *bfloat16, [num_experts, intermediate_size, hidden_size]
    result_ptr,                # *float32, [num_tokens, hidden_size] fp32 accumulation
    num_tokens: tl.constexpr,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    num_experts_per_tok: tl.constexpr,
    E: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * 128 + tl.arange(0, 128)
    mask = offs < E
    exp = tl.load(sorted_exp_ptr + offs, mask=mask, other=0)
    wt_bf16 = tl.load(flattened_wt_ptr + offs, mask=mask, other=tl.zeros((), dtype=tl.bfloat16))
    # token id = index // num_experts_per_tok
    n = offs // num_experts_per_tok

    # Load hidden state row (fp32 for numerics)
    # hidden_states_ptr has bfloat16 rows, we can load as bf16 and convert to fp32
    hs_ptrs = hidden_states_ptr + n * hidden_size + tl.arange(0, hidden_size)
    hs = tl.load(hs_ptrs, mask=mask, other=tl.zeros((hidden_size,), dtype=tl.bfloat16)).to(tl.float32)  # [hidden_size] fp32

    # Compute gate_out = hs @ expert_gate_w[exp] -> [hidden_size, intermediate_size] (fp32)
    gate_out = tl.zeros((hidden_size, intermediate_size), dtype=tl.float32)
    for j in range(0, intermediate_size):
        # Row j of expert_gate_w[exp]
        row_ptr = expert_gate_w_ptr + exp * (hidden_size * intermediate_size) + j * hidden_size + tl.arange(0, hidden_size)
        row = tl.load(row_ptr, mask=mask, other=tl.zeros((hidden_size,), dtype=tl.bfloat16)).to(tl.float32)
        gate_out[:, j] = tl.dot(hs, row)

    # up_out = hs @ expert_up_w[exp] -> [hidden_size, intermediate_size] (fp32)
    up_out = tl.zeros((hidden_size, intermediate_size), dtype=tl.float32)
    for j in range(0, intermediate_size):
        row_ptr = expert_up_w_ptr + exp * (hidden_size * intermediate_size) + j * hidden_size + tl.arange(0, hidden_size)
        row = tl.load(row_ptr, mask=mask, other=tl.zeros((hidden_size,), dtype=tl.bfloat16)).to(tl.float32)
        up_out[:, j] = tl.dot(hs, row)

    # SiLU and multiply
    silu = tl.sigmoid(gate_out) * gate_out  # SiLU(x) = x * sigmoid(x)
    activated = silu * up_out  # [hidden_size, intermediate_size] fp32

    # expert_outputs = activated @ expert_down_w[exp] -> [hidden_size] (fp32)
    expert_outputs = tl.zeros((hidden_size,), dtype=tl.float32)
    for k in range(0, hidden_size):
        acc = 0.0
        for j in range(0, intermediate_size):
            down_vec_ptr = expert_down_w_ptr + exp * (intermediate_size * hidden_size) + j * hidden_size + k
            down_val = tl.load(down_vec_ptr, mask=mask, other=0.0).to(tl.float32)
            acc += activated[k, j] * down_val
        expert_outputs[k] = acc

    # Atomic add into result[token, :]
    result_row_ptr = result_ptr + n * hidden_size + tl.arange(0, hidden_size)
    # wt_bf16 is bfloat16; convert to fp32 scalar for atomic add
    wt_fp32 = wt_bf16.to(tl.float32)
    # Loop over hidden_size and atomic_add each element
    for k in range(0, hidden_size):
        val = expert_outputs[k] * wt_fp32
        tl.atomic_add(result_row_ptr + k, val, mask=mask)


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        # Triton-only forward: no torch ops. Launch required kernels.
        assert TRITON_AVAILABLE, "Triton not available"
        device = hidden_states.device
        dtype = hidden_states.dtype

        num_tokens, hidden_size = hidden_states.shape
        num_experts, gate_h, intermediate_size = expert_gate_weights.shape
        _, up_h, up_inter = expert_up_weights.shape
        _, down_inter, down_h = expert_down_weights.shape
        assert gate_h == hidden_size and up_h == hidden_size and down_h == hidden_size, "Weight dims mismatch"
        num_experts_per_tok = selected_experts.shape[1]
        E = num_tokens * num_experts_per_tok

        # Flatten selected_experts to int32
        flattened_exp = torch.empty(E, dtype=torch.int32, device=device)
        grid_flatten_exp = (triton.cdiv(E, 128),)
        flatten_selected_exp_kernel[grid_flatten_exp](
            selected_experts, flattened_exp, num_tokens, num_experts_per_tok, E
        )

        # Flatten routing weights to bfloat16
        flattened_wt = torch.empty(E, dtype=torch.bfloat16, device=device)
        grid_flatten_wt = (triton.cdiv(E, 128),)
        flatten_weights_kernel[grid_flatten_wt](
            routing_weights, flattened_wt, num_tokens, num_experts_per_tok, E
        )

        # Stable sort by expert id (ascending), tracking positions
        to_sort = flattened_exp.clone()  # int32, shape [E]
        sorted_exp = torch.empty(E, dtype=torch.int32, device=device)
        pos = torch.empty(E, dtype=torch.int32, device=device)
        T = (E // 2) + (1 if E % 2 != 0 else 0)
        odd_even_stable_sort_experts_by_id_kernel[(E,)](
            to_sort, sorted_exp, pos, E, T
        )

        # Compute counts per expert
        counts = torch.empty(num_experts, dtype=torch.int32, device=device)
        grid_counts = (num_experts,)
        bincount_experts_kernel[grid_counts](
            sorted_exp, counts, E, num_experts
        )

        # Compute starts via cumsum (prefix sum)
        starts = torch.empty(num_experts, dtype=torch.int32, device=device)
        grid_starts = (num_experts,)
        cumsum_starts_kernel[grid_starts](
            counts, starts, num_experts
        )

        # Compute within_pos and valid mask (valid if within_pos < capacity)
        capacity_scalar = max(int((E / num_experts) * 1.25 + 0.5), 1)
        valid = torch.empty(E, dtype=torch.int32, device=device)
        grid_valid = (triton.cdiv(E, 128),)
        compute_within_pos_valid_kernel[grid_valid](
            sorted_exp, starts, capacity_scalar, valid, E
        )

        # Prepare output (fp32 accumulation) and launch full scatter weighted add
        result_fp32 = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=device)

        grid_scatter = (triton.cdiv(E, 128),)
        scatter_weighted_add_kernel[grid_scatter](
            sorted_exp, flattened_wt, hidden_states,
            expert_gate_weights, expert_up_weights, expert_down_weights,
            result_fp32,
            num_tokens, hidden_size, intermediate_size, num_experts_per_tok, E
        )

        # Return result cast to bfloat16 to match expected dtype
        return result_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
