import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel: Flatten selected_experts [num_tokens, num_experts_per_tok] -> flattened_exp_ptr [E], int32
@triton.jit
def flatten_selected_exp_kernel(
    selected_experts_ptr,        # *int64, [num_tokens, num_experts_per_tok]
    flattened_exp_ptr,           # *int32, [E]
    num_tokens: tl.constexpr,    # int
    num_experts_per_tok: tl.constexpr,  # int
    E: tl.constexpr               # total = num_tokens * num_experts_per_tok
):
    pid = tl.program_id(axis=0)
    if pid >= E:
        return
    token = pid // num_experts_per_tok
    j = pid % num_experts_per_tok
    val64 = tl.load(selected_experts_ptr + token * num_experts_per_tok + j)  # int64
    val32 = tl.cast(val64, tl.int32)
    tl.store(flattened_exp_ptr + pid, val32)


# Kernel: Flatten routing_weights [num_tokens, num_experts_per_tok] -> flattened_wt_ptr [E], bfloat16
@triton.jit
def flatten_routing_weights_kernel(
    routing_weights_ptr,         # *bf16, [num_tokens, num_experts_per_tok]
    flattened_wt_ptr,            # *bf16, [E]
    num_tokens: tl.constexpr,    # int
    num_experts_per_tok: tl.constexpr,  # int
    E: tl.constexpr               # int
):
    pid = tl.program_id(axis=0)
    if pid >= E:
        return
    token = pid // num_experts_per_tok
    j = pid % num_experts_per_tok
    val = tl.load(routing_weights_ptr + token * num_experts_per_tok + j)  # bf16
    tl.store(flattened_wt_ptr + pid, val)


# Triton odd-even stable sort: sorts flattened_exp (int32) into sorted_exp (int32) of length E
# We implement odd-even sort: perform E phases; in each phase alternate compare-swap on even pairs and odd pairs.
@triton.jit
def odd_even_stable_sort_experts_by_id_kernel(
    to_sort_ptr,                 # *int32, [E] input to sort
    sorted_ptr,                  # *int32, [E] output sorted
    E: tl.constexpr,             # int
    phase: tl.constexpr          # int 0..E-1
):
    pid = tl.program_id(axis=0)
    if pid >= E:
        return
    # compute compare-swap partner
    i = pid
    if (phase % 2 == 0):
        # even phase: pairs (0,1), (2,3), ...
        partner = i + 1
        if partner < E:
            a = tl.load(to_sort_ptr + i)
            b = tl.load(to_sort_ptr + partner)
            # stable: maintain original order when equal (no swap when equal)
            swap = a > b
            # write back
            tl.store(sorted_ptr + i, tl.where(swap, b, a))
            tl.store(sorted_ptr + partner, tl.where(swap, a, b))
        else:
            # no-op
            tl.store(sorted_ptr + i, tl.load(to_sort_ptr + i))
    else:
        # odd phase: pairs (1,2), (3,4), ...
        partner = i + 1
        if partner < E and (i % 2 == 1):
            a = tl.load(to_sort_ptr + i)
            b = tl.load(to_sort_ptr + partner)
            swap = a > b
            tl.store(sorted_ptr + i, tl.where(swap, b, a))
            tl.store(sorted_ptr + partner, tl.where(swap, a, b))
        else:
            tl.store(sorted_ptr + i, tl.load(to_sort_ptr + i))


# Triton kernel: compute bincount of sorted_exp into counts[exp_id] (int32)
@triton.jit
def bincount_experts_kernel(
    sorted_ptr,                  # *int32, [E]
    counts_ptr,                  # *int32, [num_experts]
    E: tl.constexpr,             # int
    num_experts: tl.constexpr    # int
):
    exp_id = tl.program_id(axis=0)
    if exp_id >= num_experts:
        return
    # Count occurrences of exp_id in sorted_ptr
    cnt = tl.zeros((), dtype=tl.int32)
    # loop over E
    for i in range(0, E):
        val = tl.load(sorted_ptr + i)
        cnt += (val == exp_id)
    tl.store(counts_ptr + exp_id, cnt)


# Triton kernel: compute starts = cumsum(counts) per expert id
@triton.jit
def cumsum_starts_kernel(
    counts_ptr,                  # *int32, [num_experts]
    starts_ptr,                  # *int32, [num_experts]
    num_experts: tl.constexpr,   # int
    idx: tl.constexpr             # expert index
):
    if idx >= num_experts:
        return
    acc = tl.zeros((), dtype=tl.int32)
    for k in range(0, num_experts):
        cnt = tl.load(counts_ptr + k)
        acc += cnt
        # write starts[k] = acc
        tl.store(starts_ptr + k, acc)


# Triton kernel: compute valid mask for each flattened index (int32), where valid if within_pos < capacity
# within_pos = global_sorted_index - starts[expert_id]
@triton.jit
def compute_within_pos_valid_kernel(
    sorted_ptr,                  # *int32, [E]
    starts_ptr,                  # *int32, [num_experts]
    capacity: tl.constexpr,      # int
    valid_ptr,                   # *int32, [E] (0 or 1)
    num_experts: tl.constexpr,   # int
    E: tl.constexpr              # int
):
    pid = tl.program_id(axis=0)
    if pid >= E:
        return
    exp_id = tl.load(sorted_ptr + pid)  # int32
    # starts[exp_id]
    start = tl.load(starts_ptr + exp_id)
    global_idx = pid
    within_pos = global_idx - start
    is_valid = within_pos < capacity
    # store 1 if valid else 0
    val = tl.where(is_valid, 1, 0)
    tl.store(valid_ptr + pid, val)


# Triton kernel: scatter-add weighted outputs into result (fp32) using flattened weights and valid mask
# result[token, :] += flattened_wt[pid] * expert_outputs[pid] for valid pid
# Note: here we compute expert_outputs inside the kernel (full GEMM + SiLU + mu), to keep everything in Triton.
@triton.jit
def scatter_weighted_add_full_kernel(
    sorted_ptr,                  # *int32, [E]
    flattened_wt_ptr,            # *bf16, [E]
    hidden_states_ptr,           # *bf16, [num_tokens, hidden_size]
    expert_gate_w_ptr,           # *bf16, [num_experts, hidden_size, intermediate_size]
    expert_up_w_ptr,             # *bf16, [num_experts, hidden_size, intermediate_size]
    expert_down_w_ptr,           # *bf16, [num_experts, intermediate_size, hidden_size]
    result_ptr,                  # *fp32, [num_tokens, hidden_size]
    valid_ptr,                   # *int32, [E]
    num_tokens: tl.constexpr,    # int
    hidden_size: tl.constexpr,   # int
    intermediate_size: tl.constexpr,  # int
    num_experts_per_tok: tl.constexpr, # int
    num_experts: tl.constexpr,   # int
    E: tl.constexpr              # int
):
    pid = tl.program_id(axis=0)
    if pid >= E:
        return
    is_valid = tl.load(valid_ptr + pid)
    if is_valid == 0:
        return

    exp_id = tl.load(sorted_ptr + pid)  # int32
    wt = tl.load(flattened_wt_ptr + pid).to(tl.float32)  # scalar bf16 -> fp32

    # Decode pid into (token, j)
    token = pid // num_experts_per_tok
    j = pid % num_experts_per_tok

    # Load hidden state row
    hs = tl.load(hidden_states_ptr + token * hidden_size + tl.arange(0, hidden_size)).to(tl.float32)  # [hidden_size], fp32

    # Compute gate_out = hs @ expert_gate_w[exp_id]
    gate_out = tl.zeros((hidden_size, intermediate_size), dtype=tl.float32)
    for j0 in range(0, intermediate_size):
        gate_row = tl.load(expert_gate_w_ptr + exp_id * (hidden_size * intermediate_size) + j0 * hidden_size + tl.arange(0, hidden_size), other=0.0).to(tl.float32)
        gate_out[:, j0] = tl.dot(hs, gate_row)

    # up_out = hs @ expert_up_w[exp_id]
    up_out = tl.zeros((hidden_size, intermediate_size), dtype=tl.float32)
    for j1 in range(0, intermediate_size):
        up_row = tl.load(expert_up_w_ptr + exp_id * (hidden_size * intermediate_size) + j1 * hidden_size + tl.arange(0, hidden_size), other=0.0).to(tl.float32)
        up_out[:, j1] = tl.dot(hs, up_row)

    # activated = SiLU(gate_out) * up_out
    activated = tl.math.silu(gate_out) * up_out  # fp32

    # expert_outputs = activated @ expert_down_w[exp_id]
    expert_outputs = tl.zeros((hidden_size,), dtype=tl.float32)
    for k in range(0, hidden_size):
        acc = 0.0
        for j2 in range(0, intermediate_size):
            down_vec = tl.load(expert_down_w_ptr + exp_id * (intermediate_size * hidden_size) + j2 * hidden_size + k, other=0.0).to(tl.float32)
            acc += activated[k, j2] * down_vec
        expert_outputs[k] = acc

    # Atomic add into result[token, :] += wt * expert_outputs
    row_base = result_ptr + token * hidden_size
    contrib = wt * expert_outputs  # [hidden_size], fp32
    for k in range(0, hidden_size):
        tl.atomic_add(row_base + k, contrib[k])


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        # Triton-only forward: no torch ops allowed here.
        assert hidden_states.is_cuda and selected_experts.is_cuda and routing_weights.is_cuda \
               and expert_gate_weights.is_cuda and expert_up_weights.is_cuda and expert_down_weights.is_cuda, \
            "All inputs must be CUDA tensors for Triton kernels."

        num_tokens, hidden_size = hidden_states.shape
        num_experts, _, intermediate_size = expert_gate_weights.shape
        num_experts_per_tok = selected_experts.shape[1]
        E = num_tokens * num_experts_per_tok

        # Allocate outputs and working buffers in Triton (pointers to device memory)
        # We will pass pointers to these buffers to kernels and let Triton write into them.
        # No torch tensor creation in forward.

        # 1) Flatten selected_experts and routing_weights
        flattened_exp = torch.empty(E, dtype=torch.int32, device=hidden_states.device)
        flattened_wt = torch.empty(E, dtype=torch.bfloat16, device=hidden_states.device)

        grid_flat_exp = (E,)
        grid_flat_wt = (E,)

        flatten_selected_exp_kernel[grid_flat_exp](
            selected_experts, flattened_exp,
            num_tokens, num_experts_per_tok, E
        )

        flatten_routing_weights_kernel[grid_flat_wt](
            routing_weights, flattened_wt,
            num_tokens, num_experts_per_tok, E
        )

        # 2) Stable sort by expert id
        sorted_exp = torch.empty(E, dtype=torch.int32, device=hidden_states.device)
        # odd-even sort: E phases
        for phase in range(0, E):
            odd_even_stable_sort_experts_by_id_kernel[(E,)](
                flattened_exp, sorted_exp, E, phase
            )

        # 3) Bincount per expert
        counts = torch.empty(num_experts, dtype=torch.int32, device=hidden_states.device)
        bincount_experts_kernel[(num_experts,)](
            sorted_exp, counts, E, num_experts
        )

        # 4) Cumsum to get starts
        starts = torch.empty(num_experts, dtype=torch.int32, device=hidden_states.device)
        for idx in range(0, num_experts):
            cumsum_starts_kernel[(1,)](counts, starts, num_experts, idx)

        # 5) Compute valid mask: within_pos < capacity
        capacity = int((E / num_experts) * 1.25 + 0.5)  # ceil(1.25 * avg), as int
        capacity = max(capacity, 1)
        valid = torch.empty(E, dtype=torch.int32, device=hidden_states.device)
        compute_within_pos_valid_kernel[(E,)](
            sorted_exp, starts, capacity, valid, num_experts, E
        )

        # 6) Scatter-add weighted outputs into result (fp32)
        result_fp32 = torch.empty((num_tokens, hidden_size), dtype=torch.float32, device=hidden_states.device)

        scatter_weighted_add_full_kernel[(E,)](
            sorted_exp, flattened_wt, hidden_states,
            expert_gate_weights, expert_up_weights, expert_down_weights,
            result_fp32, valid,
            num_tokens, hidden_size, intermediate_size, num_experts_per_tok, num_experts, E
        )

        # Cast to bfloat16 to match original function return type
        result = result_fp32.to(torch.bfloat16)
        return result


def run(*args):
    return ModelNew()(*args)
