import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 0: Flatten selected_experts (int64 -> int32) and store into 1D array.
@triton.jit
def flatten_experts_kernel(
    src_ptr,           # *int64, shape [num_tokens, num_experts_per_tok]
    dst_exp_ptr,       # *int32, shape [E]
    num_tokens: tl.constexpr,
    num_experts_per_tok: tl.constexpr,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < E
    val = tl.load(src_ptr + offsets, mask=mask, other=0)  # int64
    val = val.to(tl.int32)
    tl.store(dst_exp_ptr + offsets, val, mask=mask)


# Kernel 1: Flatten routing_weights (bf16 -> bf16) into 1D array.
@triton.jit
def flatten_weights_kernel(
    src_ptr,           # *bf16, shape [num_tokens, num_experts_per_tok]
    dst_wt_ptr,        # *bf16, shape [E]
    num_tokens: tl.constexpr,
    num_experts_per_tok: tl.constexpr,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < E
    val = tl.load(src_ptr + offsets, mask=mask, other=tl.zeros((), dtype=tl.bfloat16))
    tl.store(dst_wt_ptr + offsets, val, mask=mask)


# Kernel 2: Odd-even stable sort by expert id (int32). Produces sorted_exp_ptr.
@triton.jit
def odd_even_stable_sort_experts_by_id_kernel(
    to_sort_ptr,           # *int32, shape [E] values to sort by id
    sorted_exp_ptr,        # *int32, shape [E] output sorted by id
    E: tl.constexpr,
    BLOCK: tl.constexpr,   # unused in this scalar kernel
):
    # Perform odd-even sort over E elements. After ~E phases, list is sorted.
    for phase in range(0, 1024):
        # Even phase: compare (0,1), (2,3), ...
        for i in range(0, E, 2):
            if (i + 1) < E:
                a = tl.load(to_sort_ptr + i)
                b = tl.load(to_sort_ptr + i + 1)
                if a > b:
                    tl.store(to_sort_ptr + i, b)
                    tl.store(to_sort_ptr + i + 1, a)
        # Odd phase: compare (1,2), (3,4), ...
        for i in range(1, E, 2):
            if (i + 1) < E:
                a = tl.load(to_sort_ptr + i)
                b = tl.load(to_sort_ptr + i + 1)
                if a > b:
                    tl.store(to_sort_ptr + i, b)
                    tl.store(to_sort_ptr + i + 1, a)

    # Copy sorted ids to output
    for i in range(0, E):
        sid = tl.load(to_sort_ptr + i)
        tl.store(sorted_exp_ptr + i, sid)


# Kernel 3: Compute per-expert counts (bincount) of sorted_exp.
@triton.jit
def bincount_experts_kernel(
    exp_ptr,               # *int32, shape [E]
    counts_ptr,            # *int32, shape [num_experts]
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Simple loop over E to count each expert id
    for i in range(0, E):
        eid = tl.load(exp_ptr + i)
        tl.atomic_add(counts_ptr + eid, 1)


# Kernel 4: Compute per-expert starts via cumsum (no torch.cumsum).
@triton.jit
def cumsum_starts_kernel(
    counts_ptr,            # *int32, shape [num_experts]
    starts_ptr,            # *int32, shape [num_experts]
    num_experts: tl.constexpr,
    BLOCK: tl.constexpr,   # can be 1
):
    acc = 0
    for i in range(0, num_experts):
        cnt = tl.load(counts_ptr + i)
        acc += cnt
        tl.store(starts_ptr + i, acc)


# Kernel 5: Compute within_pos for each flattened index i:
# within_pos = i - starts[sorted_exp[i]]; valid if within_pos < capacity.
# Store 0/1 in valid_ptr[i] (int32).
@triton.jit
def compute_within_pos_valid_kernel(
    sorted_exp_ptr,        # *int32, shape [E]
    starts_ptr,            # *int32, shape [num_experts]
    capacity,              # int32 scalar
    valid_ptr,             # *int32, shape [E]
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for i in range(0, E):
        sid = tl.load(sorted_exp_ptr + i)
        start = tl.load(starts_ptr + sid)
        within = i - start
        is_valid = within < capacity
        tl.store(valid_ptr + i, is_valid.to(tl.int32))


# Kernel 6: Scatter-add weighted outputs into result. For each flattened index i:
#   - expert_id = sorted_exp[i]
#   - token_id = i // num_experts_per_tok
#   - weight = flattened_wt[i]
#   - hidden_state_row = hidden_states[token_id]
#   - Compute gate_out, up_out, SiLU, multiply, then expert_outputs = activated @ down, and result[token_id] += weight * expert_outputs (atomic_add in bf16).
@triton.jit
def scatter_weighted_add_kernel(
    sorted_exp_ptr,        # *int32, shape [E]
    flattened_wt_ptr,      # *bf16, shape [E]
    hidden_states_ptr,     # *bf16, shape [num_tokens, hidden_size], contiguous
    expert_gate_w_ptr,     # *bf16, shape [num_experts, hidden_size*intermediate_size], contiguous
    expert_up_w_ptr,       # *bf16, shape [num_experts, hidden_size*intermediate_size], contiguous
    expert_down_w_ptr,     # *bf16, shape [num_experts, intermediate_size*hidden_size], contiguous
    result_ptr,            # *bf16, shape [num_tokens, hidden_size], contiguous
    num_tokens: tl.constexpr,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    num_experts_per_tok: tl.constexpr,
    E: tl.constexpr,
):
    for i in range(0, E):
        # Load expert id and weight
        exp_i = tl.load(sorted_exp_ptr + i)             # int32
        wt = tl.load(flattened_wt_ptr + i)              # bf16
        # Compute token id from flattened index: n = i // num_experts_per_tok
        n = i // num_experts_per_tok                    # int32 scalar

        # Load hidden state row as bf16, compute in fp32
        hs_row = tl.load(hidden_states_ptr + n * hidden_size + tl.arange(0, hidden_size), mask=True, other=tl.zeros((hidden_size,), dtype=tl.bfloat16))
        hs = hs_row.to(tl.float32)  # [hidden_size], fp32

        # Compute gate_out = hs @ expert_gate_w[exp_i] -> [hidden_size, intermediate_size] (fp32)
        gate_out = tl.zeros((hidden_size, intermediate_size), dtype=tl.float32)
        for j in range(0, intermediate_size):
            gate_row = tl.load(expert_gate_w_ptr + exp_i * (hidden_size * intermediate_size) + j * hidden_size + tl.arange(0, hidden_size),
                               mask=True, other=tl.zeros((hidden_size,), dtype=tl.bfloat16)).to(tl.float32)
            gate_out[:, j] = tl.dot(hs, gate_row)  # [hidden_size]

        # up_out = hs @ expert_up_w[exp_i]
        up_out = tl.zeros((hidden_size, intermediate_size), dtype=tl.float32)
        for j in range(0, intermediate_size):
            up_row = tl.load(expert_up_w_ptr + exp_i * (hidden_size * intermediate_size) + j * hidden_size + tl.arange(0, hidden_size),
                             mask=True, other=tl.zeros((hidden_size,), dtype=tl.bfloat16)).to(tl.float32)
            up_out[:, j] = tl.dot(hs, up_row)

        # SiLU and multiply
        activated = tl.silu(gate_out) * up_out  # fp32

        # expert_outputs = activated @ expert_down_w[exp_i] -> [hidden_size]
        expert_outputs = tl.zeros((hidden_size,), dtype=tl.float32)
        for k in range(0, hidden_size):
            acc = 0.0
            for j in range(0, intermediate_size):
                down_vec = tl.load(expert_down_w_ptr + exp_i * (intermediate_size * hidden_size) + j * hidden_size + tl.arange(0, hidden_size),
                                   mask=True, other=tl.zeros((hidden_size,), dtype=tl.bfloat16)).to(tl.float32)
                prod = activated[k, j] * down_vec  # [hidden_size]
                acc += tl.sum(prod)
            expert_outputs[k] = acc

        # Convert weight to fp32
        wt_f32 = wt.to(tl.float32)

        # Atomic add into result for token n (cast to bf16 before atomic_add)
        # result[n, :] += wt * expert_outputs
        for k in range(0, hidden_size):
            res_val = tl.load(result_ptr + n * hidden_size + k, mask=True, other=tl.zeros((), dtype=tl.bfloat16))
            res_val_f32 = res_val.to(tl.float32)
            new_val_f32 = res_val_f32 + wt_f32 * expert_outputs[k]
            new_val_bf16 = new_val_f32.to(tl.bfloat16)
            tl.atomic_add(result_ptr + n * hidden_size + k, new_val_bf16)


# ModelNew entry point
class ModelNew(nn.Module):
    def forward(self, hidden_states, selected_experts, routing_weights,
                expert_gate_weights, expert_up_weights, expert_down_weights):
        # Ensure CUDA device and bfloat16 dtype (as in the original)
        device = hidden_states.device
        assert hidden_states.is_cuda, "ModelNew.forward requires CUDA tensors"
        assert hidden_states.dtype == torch.bfloat16, "hidden_states must be bfloat16"

        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        num_experts, _, intermediate_size = expert_gate_weights.shape
        num_experts_per_tok = selected_experts.shape[1]
        E = num_tokens * num_experts_per_tok

        # Allocate flattened arrays
        flattened_exp = torch.empty(E, dtype=torch.int32, device=device)
        flattened_wt = torch.empty(E, dtype=torch.bfloat16, device=device)

        # Launch flatten kernels
        BLOCK = 1024
        grid_exp = (triton.cdiv(E, BLOCK),)
        flatten_experts_kernel[grid_exp](
            selected_experts, flattened_exp, num_tokens, num_experts_per_tok, E, BLOCK
        )
        grid_wt = (triton.cdiv(E, BLOCK),)
        flatten_weights_kernel[grid_wt](
            routing_weights, flattened_wt, num_tokens, num_experts_per_tok, E, BLOCK
        )

        # Perform stable sort by expert id (odd-even sort)
        to_sort = flattened_exp.clone()  # int32, shape [E]
        sorted_exp = torch.empty(E, dtype=torch.int32, device=device)
        odd_even_stable_sort_experts_by_id_kernel[(E,)](
            to_sort, sorted_exp, E, 1
        )

        # Compute counts per expert
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        bincount_experts_kernel[(num_experts,)](sorted_exp, counts, E, 1)

        # Compute starts via cumsum
        starts = torch.empty(num_experts, dtype=torch.int32, device=device)
        cumsum_starts_kernel[(num_experts,)](counts, starts, num_experts, 1)

        # Compute within_pos and valid mask (not strictly needed for scatter, but kept for completeness)
        valid = torch.empty(E, dtype=torch.int32, device=device)
        capacity_scalar = int((E / num_experts) * 1.25 + 0.5)  # ceil(1.25 * avg)
        capacity_scalar = max(capacity_scalar, 1)
        compute_within_pos_valid_kernel[(E,)](
            sorted_exp, starts, capacity_scalar, valid, E, 1
        )

        # Prepare output and scatter-add
        result = torch.zeros(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)

        # Launch scatter kernel (does all the compute and atomic add)
        scatter_weighted_add_kernel[(E,)](
            sorted_exp, flattened_wt, hidden_states,
            expert_gate_weights, expert_up_weights, expert_down_weights,
            result,
            num_tokens, hidden_size, intermediate_size, num_experts_per_tok, E, 1
        )

        return result


def run(*args):
    return ModelNew()(*args)
